import os
import shutil
import tarfile
import numpy as np
import librosa
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
# import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import sys

DATASET_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
TAR_FILENAME = "train-clean-100.tar.gz"
DATA_DIR = './data'
H5_FILE = './data'
EXTRACT_PATH = os.path.join(DATA_DIR, 'train-clean-100_extracted', 'LibriSpeech', 'train-clean-100')

MAX_AUDIO_LEN_S = 10.0
SEGMENT_LEN_S = 2.0
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

def analyze_dataset():
    """Analizira osnovne lastnosti podatkovne zbirke."""
    if not os.path.exists(EXTRACT_PATH):
        print(f"Mapa {EXTRACT_PATH} ne obstaja. Prosim, najprej zaženite download_and_extract_data().")
        return

    speaker_dirs = [d for d in os.listdir(EXTRACT_PATH) if os.path.isdir(os.path.join(EXTRACT_PATH, d))]

    speaker_stats = {}
    total_files = 0

    for speaker_id in speaker_dirs:
        speaker_path = os.path.join(EXTRACT_PATH, speaker_id)
        for root, _, files in os.walk(speaker_path):
            flac_files = [f for f in files if f.endswith(".flac")]
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {'count': 0, 'durations': []}

            speaker_stats[speaker_id]['count'] += len(flac_files)
            total_files += len(flac_files)

            for f in flac_files:
                try:
                    file_path = os.path.join(root, f)
                    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    speaker_stats[speaker_id]['durations'].append(librosa.get_duration(y=y, sr=sr))
                except Exception as e:
                    print(f"Napaka pri branju datoteke {file_path}: {e}")

    num_speakers = len(speaker_stats)
    if num_speakers == 0:
        print("V podatkovni zbirki ni bilo najdenih govorcev. Preverite pot do podatkov.")
        return

    file_counts = [s['count'] for s in speaker_stats.values()]
    all_durations = [d for s in speaker_stats.values() for d in s['durations']]

    print("--- Analiza Podatkovne Zbirke ---")
    print(f"Število govorcev: {num_speakers}")
    print(f"Število vseh posnetkov: {total_files}")
    print("\nStatistika števila posnetkov na govorca:")
    print(f"  - Povprečje: {np.mean(file_counts):.2f}")
    print(f"  - Std odklon: {np.std(file_counts):.2f}")
    print(f"  - Minimum: {np.min(file_counts)}")
    print(f"  - Maksimum: {np.max(file_counts)}")

    print("\nStatistika dolžine posnetkov (s):")
    print(f"  - Povprečje: {np.mean(all_durations):.2f}")
    print(f"  - Std odklon: {np.std(all_durations):.2f}")
    print(f"  - Minimum: {np.min(all_durations):.2f}")
    print(f"  - Maksimum: {np.max(all_durations):.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(file_counts, bins=30)
    plt.title("Porazdelitev št. posnetkov na govorca")
    plt.xlabel("Število posnetkov")
    plt.ylabel("Število govorcev")

    plt.subplot(1, 2, 2)
    sns.histplot(all_durations, bins=30)
    plt.title("Porazdelitev dolžin posnetkov")
    plt.xlabel("Dolžina (s)")
    plt.ylabel("Število posnetkov")
    plt.tight_layout()
    plt.show()

# # --- 2. Predobdelava in ustvarjanje HDF5 datoteke ---

def process_and_save_features(num_speakers_to_process=100):
    if os.path.exists(H5_FILE):
        print(f"H5 datoteka {H5_FILE} že obstaja. Brisanje in ponovno ustvarjanje.")
        os.remove(H5_FILE)

    features = []
    labels = []

    all_speaker_dirs = sorted([d for d in os.listdir(EXTRACT_PATH) if os.path.isdir(os.path.join(EXTRACT_PATH, d))])
    speakers_to_process = all_speaker_dirs[:num_speakers_to_process]
    print(f"Obdelujem {len(speakers_to_process)} govorcev...")

    segment_len_samples = int(SEGMENT_LEN_S * SAMPLE_RATE)
    max_len_samples = int(MAX_AUDIO_LEN_S * SAMPLE_RATE)

    for speaker_id in speakers_to_process:
        speaker_path = os.path.join(EXTRACT_PATH, speaker_id)
        print(f"Obdelujem govorca: {speaker_id}")
        for root, _, files in os.walk(speaker_path):
            for f in files:
                if not f.endswith(".flac"):
                    continue

                file_path = os.path.join(root, f)
                try:
                    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    if len(y) > max_len_samples:
                        y = y[:max_len_samples]

                    for start in range(0, len(y) - segment_len_samples + 1, segment_len_samples):
                        segment = y[start:start + segment_len_samples]
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                        features.append(mfcc)
                        labels.append(speaker_id)

                except Exception as e:
                    print(f"Napaka pri datoteki {file_path}: {e}")

    with h5py.File(H5_FILE, 'w') as hf:
        hf.create_dataset('features', data=np.array(features, dtype=np.float32))
        labels_ascii = [n.encode("ascii", "ignore") for n in labels]
        hf.create_dataset('labels', data=np.array(labels_ascii))

    print(f"Končano. Shranjenih {len(features)} vzorcev.")


def load_data_from_h5():
    """Naloži podatke iz HDF5 in jih pripravi za učenje."""
    if not os.path.exists(H5_FILE):
        print(f"Datoteka {H5_FILE} ne obstaja. Prosim, najprej zaženite process_and_save_features().")
        return None, None, None, None, None

    with h5py.File(H5_FILE, 'r') as hf:
        X = np.array(hf.get('features'))
        y_str = np.array(hf.get('labels'))

    y_str = [label.decode('utf-8') for label in y_str]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42,
                                                        stratify=y_categorical)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    print(f"Velikost učne množice: {X_train.shape[0]}")
    print(f"Velikost validacijske množice: {X_val.shape[0]}")
    print(f"Velikost testne množice: {X_test.shape[0]}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, num_classes


# --- 3. Definicija in učenje modela ---

def create_model(input_shape, num_classes, layers_config='base', size_config='base'):
    """Ustvari Keras model glede na konfiguracijo."""
    # Konfiguracija velikosti
    if size_config == 'smaller':
        filters = [16, 32]
        dense_units = 64
    elif size_config == 'larger':
        filters = [64, 128]
        dense_units = 256
    else:  # base
        filters = [32, 64]
        dense_units = 128

    model = Sequential()
    model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Konfiguracija arhitekture
    if layers_config == 'base' or layers_config == 'deep':
        model.add(Conv2D(filters[1], (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

    if layers_config == 'deep':
        model.add(Conv2D(128, (3, 3), activation='relu'))  # Dodatna plast
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history, title_suffix=""):
    """Izriše grafe natančnosti in izgube."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f"Metrike učenja - {title_suffix}", fontsize=16)

    axs[0].plot(history.history['accuracy'], label='Učna natančnost')
    axs[0].plot(history.history['val_accuracy'], label='Validacijska natančnost')
    axs[0].set_ylabel('Natančnost')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Natančnost med učenjem')

    axs[1].plot(history.history['loss'], label='Učna izguba')
    axs[1].plot(history.history['val_loss'], label='Validacijska izguba')
    axs[1].set_xlabel('Epohe')
    axs[1].set_ylabel('Izguba')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Izguba med učenjem')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def add_noise(data, snr_db):
    """Doda šum (normalno porazdeljen) podatkovni množici z določenim SNR."""
    data_noisy = np.copy(data)
    for i in range(len(data_noisy)):
        signal = data_noisy[i]
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        data_noisy[i] = signal + noise
    return data_noisy


def run_experiment(exp_title, model_config, noise_snr=None):
    """Glavna funkcija za izvedbo eksperimenta."""
    print(f"\n{'=' * 20} ZAČETEK EKSPERIMENTA: {exp_title} {'=' * 20}")

    # 1. Naloži podatke
    (X_train, y_train), (X_val, y_val), (X_test, y_test), le, num_classes = load_data_from_h5()
    if X_train is None: return

    input_shape = X_train.shape[1:]

    # 2. Gradnja in učenje modela
    model = create_model(input_shape, num_classes, **model_config)
    print("\n--- Arhitektura modela ---")
    model.summary()

    # Callbacki
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'best_model_{exp_title}.h5', save_best_only=True, monitor='val_loss')

    start_time = time.time()

    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    training_time = time.time() - start_time
    print(f"\nČas učenja: {training_time:.2f} sekund ({training_time / 60:.2f} minut)")

    # 3. Evalvacija in poročanje
    best_model = load_model(f'best_model_{exp_title}.h5')

    # Priprava testnih podatkov (z morebitnim šumom)
    eval_X_test = X_test
    if noise_snr is not None:
        print(f"Dodajam šum na testno množico: SNR = {noise_snr} dB")
        eval_X_test = add_noise(X_test, noise_snr)

    test_loss, test_acc = best_model.evaluate(eval_X_test, y_test, verbose=0)
    print(f"\nRezultati na testni množici ({exp_title}):")
    print(f"  - Izguba: {test_loss:.4f}")
    print(f"  - Natančnost: {test_acc * 100:.2f}%")

    plot_history(history, exp_title)

    predictions = best_model.predict(eval_X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(12, 10))
    num_to_display = min(num_classes, 15)
    sns.heatmap(conf_matrix[:num_to_display, :num_to_display], annot=True, fmt='d',
                xticklabels=le.classes_[:num_to_display], yticklabels=le.classes_[:num_to_display])
    plt.title(f'Matrika zamešitev - {exp_title} (prvih {num_to_display} govorcev)')
    plt.xlabel('Predviden razred')
    plt.ylabel('Dejanski razred')
    plt.show()


# --- Glavni zagon ---
if __name__ == '__main__':
    # --- KORAK 1: Priprava podatkov (zaženite enkrat) ---
    # analyze_dataset()

    # --- KORAK 2: Predobdelava (zaženite enkrat za določeno št. govorcev) ---
    # N_SPEAKERS = 100
    # process_and_save_features(num_speakers_to_process=N_SPEAKERS)

    # --- KORAK 3: Izvedba eksperimentov ---
    # Odkomenirajte in poženite posamezne eksperimente.

    # A. Vpliv števila plasti
    run_experiment("A1_Plasti_Simple", model_config={'layers_config': 'simple', 'size_config': 'base'})
    run_experiment("A2_Plasti_Base", model_config={'layers_config': 'base', 'size_config': 'base'})
    run_experiment("A3_Plasti_Deep", model_config={'layers_config': 'deep', 'size_config': 'base'})

    # B. Vpliv velikosti plasti (na najboljši arhitekturi iz koraka A)
    run_experiment("B1_Velikost_Smaller", model_config={'layers_config': 'base', 'size_config': 'smaller'})
    run_experiment("B2_Velikost_Base", model_config={'layers_config': 'base', 'size_config': 'base'})
    run_experiment("B3_Velikost_Larger", model_config={'layers_config': 'base', 'size_config': 'larger'})

    # C. Vpliv šuma (na najboljšem modelu iz koraka B)
    # Najprej naučite osnovni model brez šuma (npr. B2). Potem lahko samo testirate z dodanim šumom,
    # ali pa za vsak primer naučite posebej, če želite preveriti robustnost učenja.
    # Spodnji klici predvidevajo ponovno učenje za vsak primer.
    # Za evalvacijo na obstoječem modelu bi morali kodo rahlo prilagoditi.
    run_experiment("C1_Sum_Brez", model_config={'layers_config': 'base', 'size_config': 'base'}, noise_snr=None)
    run_experiment("C2_Sum_SNR20", model_config={'layers_config': 'base', 'size_config': 'base'}, noise_snr=20)
    run_experiment("C3_Sum_SNR10", model_config={'layers_config': 'base', 'size_config': 'base'}, noise_snr=10)
    run_experiment("C4_Sum_SNR0", model_config={'layers_config': 'base', 'size_config': 'base'}, noise_snr=0)

    # D. Vpliv števila govorcev
    # Za ta eksperiment morate najprej zagnati `process_and_save_features` z različnimi vrednostmi.
    # Primer:
    print("Zaženite naslednje korake ročno z odkomentiranjem:")
    print("1. process_and_save_features(num_speakers_to_process=50)")
    print("2. run_experiment('D1_Govorci_50', ...)")
    print("3. process_and_save_features(num_speakers_to_process=100)")
    print("4. run_experiment('D2_Govorci_100', ...)")