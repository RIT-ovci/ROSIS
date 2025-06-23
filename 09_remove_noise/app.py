import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. KONFIGURACIJA IN PARAMETRI ---
# Preverimo, ali je na voljo CUDA (GPU), sicer uporabimo CPE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uporabljena naprava: {DEVICE}")

# Parametri za avdio obdelavo
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 256  # Velikost spektrograma (višina)
CHUNK_SIZE = 128  # Širina segmenta spektrograma za vhod v model

# Parametri za učenje
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 15  # Prilagodite glede na čas in želeno natančnost (15-30 epoh je dober začetek)

# Poti do podatkovnih zbirk
CLEAN_SPEECH_PATH = 'datasets/LibriSpeech/dev-clean-2'
NOISE_PATH = 'datasets/UrbanSound8K'

# Mapa za shranjevanje rezultatov
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. PRIPRAVA PODATKOV (Dataset in DataLoader) ---

def load_audio_files(path, pattern='**/*.flac'):
    """Rekurzivno naloži poti do avdio datotek."""
    return glob.glob(os.path.join(path, pattern), recursive=True)



# --- 3. DEFINICIJA MODELA (U-Net) ---

class UNet(nn.Module):
    """
    Poenostavljena U-Net arhitektura za odstranjevanje šuma iz spektrogramov.
    """

    def __init__(self):
        super(UNet, self).__init__()
        # Enkoder del (pot navzdol)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Srednji del (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Dekoder del (pot navzgor)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 (iz up) + 32 (iz skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16 (iz up) + 16 (iz skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Izhodna plast
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Enkoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Dekoder s "skip connections"
        d1 = self.upconv1(b)
        d1 = torch.cat((d1, e2), dim=1)  # Skip connection
        d1 = self.dec1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, e1), dim=1)  # Skip connection
        d2 = self.dec2(d2)

        output = self.out_conv(d2)
        return output


# --- 4. CIKEL UČENJA ---

def train(model, dataloader, criterion, optimizer, device):
    """Funkcija za izvedbo ene epohe učenja."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Učenje", colour="green")

    for noisy_spec, clean_spec in progress_bar:
        noisy_spec = noisy_spec.to(device)
        clean_spec = clean_spec.to(device)

        # Ponastavi gradiente
        optimizer.zero_grad()

        # Vhod skozi model
        predicted_spec = model(noisy_spec)

        # Izračun izgube
        loss = criterion(predicted_spec, clean_spec)

        # Povratna propagacija
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


# --- 5. EVALUACIJA IN VIZUALIZACIJA ---

def denoise_and_visualize(model, clean_audio_path, noise_audio_path, output_dir, sr=SAMPLE_RATE):
    """
    Funkcija, ki vzame čist in zašumljen posnetek, ga očisti z modelom
    in shrani primerjalne spektrograme ter zvočne datoteke.
    """
    model.eval()  # Preklopi model v evalvacijski način

    # Naloži avdio
    clean_audio, _ = librosa.load(clean_audio_path, sr=sr)
    noise_audio, _ = librosa.load(noise_audio_path, sr=sr)

    # Pripravi šum
    if len(noise_audio) < len(clean_audio):
        noise_audio = np.tile(noise_audio, len(clean_audio) // len(noise_audio) + 1)
    noise_audio = noise_audio[:len(clean_audio)]

    # Ustvari zašumljen posnetek (s fiksnim SNR za primerjavo)
    snr_db = 2.5
    clean_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noise_audio ** 2)
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    noisy_audio = clean_audio + noise_audio * scale

    # Pridobi spektrograme
    clean_stft = librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    clean_mag, _ = librosa.magphase(clean_stft)
    clean_mag_db = librosa.amplitude_to_db(clean_mag, ref=np.max)

    noisy_stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag, noisy_phase = librosa.magphase(noisy_stft)
    noisy_mag_db = librosa.amplitude_to_db(noisy_mag, ref=np.max)

    # Priprava za model
    noisy_spec_tensor = torch.tensor(noisy_mag_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Napoved modela
    with torch.no_grad():
        predicted_spec_db = model(noisy_spec_tensor).squeeze(0).squeeze(0).cpu().numpy()

    # Rekonstrukcija avdia iz napovedanega spektrograma
    predicted_mag = librosa.db_to_amplitude(predicted_spec_db, ref=np.max)
    reconstructed_stft = predicted_mag * noisy_phase
    reconstructed_audio = librosa.istft(reconstructed_stft, hop_length=HOP_LENGTH, length=len(clean_audio))

    # Shranjevanje zvočnih datotek
    basename = os.path.splitext(os.path.basename(clean_audio_path))[0]
    sf.write(os.path.join(output_dir, f"{basename}_clean.wav"), clean_audio, sr)
    sf.write(os.path.join(output_dir, f"{basename}_noisy.wav"), noisy_audio, sr)
    sf.write(os.path.join(output_dir, f"{basename}_denoised.wav"), reconstructed_audio, sr)

    # Vizualizacija spektrogramov
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)

    librosa.display.specshow(clean_mag_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', ax=axes[0])
    axes[0].set_title('Originalni Čisti Spektrogram')
    axes[0].set_xlabel('')

    librosa.display.specshow(noisy_mag_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Zašumljen Spektrogram')
    axes[1].set_xlabel('')

    img = librosa.display.specshow(predicted_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='log',
                                   ax=axes[2])
    axes[2].set_title('Očiščen Spektrogram (Napoved modela)')
    axes[2].set_xlabel('Čas (s)')

    fig.colorbar(img, ax=axes, format='%+2.0f dB', label='Glasnost (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_spectrograms.png"))
    plt.show()


# --- 6. GLAVNI DEL PROGRAMA ---

# --- 6. GLAVNI DEL PROGRAMA ---

if __name__ == '__main__':
    # Naloži poti do datotek
    print("Nalaganje poti do avdio datotek...")
    clean_files = load_audio_files(CLEAN_SPEECH_PATH, pattern='**/*.flac')
    noise_files = load_audio_files(NOISE_PATH, pattern='**/fold*/*.wav')
    print(f"Najdenih {len(clean_files)} čistih posnetkov in {len(noise_files)} posnetkov šuma.")

    # Razdeli na učno in testno množico
    train_clean_files = clean_files[:-20]  # Uporabimo vse razen zadnjih 20 za učenje
    test_clean_files = clean_files[-20:]  # Zadnjih 20 datotek za evaluacijo

    # Priprava DataLoaderja
    train_dataset = DenoisingDataset(train_clean_files, noise_files, N_FFT, HOP_LENGTH, CHUNK_SIZE)

    # === GLAVNI POPRAVEK JE TUKAJ ===
    # Nastavitev num_workers=0 izklopi vzporedno procesiranje, kar prepreči segmentation fault.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Inicializacija modela, funkcije izgube in optimizatorja
    model = UNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nZačetek učenja modela...")
    train_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoha {epoch}/{NUM_EPOCHS} ---")
        avg_loss = train(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(avg_loss)
        print(f"Povprečna izguba v epohi {epoch}: {avg_loss:.4f}")

    # Shranjevanje naučenega modela
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'denoising_unet_model.pth'))
    print("\nModel shranjen.")

    # Graf učenja (izguba skozi epohe)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, '-o')
    plt.title('Učna Krivulja - Izguba skozi epohe')
    plt.xlabel('Epoha')
    plt.ylabel('MSE Izguba')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curve.png'))
    plt.show()

    # Evaluacija in vizualizacija na nekaj testnih primerih
    print("\nGeneriranje primerov in vizualizacij...")
    # Uporabimo naključen šum za vsak testni primer
    for test_file in test_clean_files:
        noise_file = np.random.choice(noise_files)
        denoise_and_visualize(model, test_file, noise_file, OUTPUT_DIR)

    print(f"\nKončano! Rezultati so shranjeni v mapi '{OUTPUT_DIR}'.")