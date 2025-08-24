import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda/nvvm/libdevice/libdevice.10.bc"

def create_dataset(speaker_ids, all_segments_dict, mapping):
    """
    Ustvari (X, y) pare (spektrogrami, oznake) za podano množico govorcev.
    """
    X, y = [], []
    for speaker_id in speaker_ids:
        label = mapping[speaker_id]
        for segment in all_segments_dict[speaker_id]:
            mel_spec = extract_mel_spectrogram(segment)
            X.append(mel_spec)
            y.append(label)
    # Pretvori v numpy array in preveri obliko
    X = np.array(X)
    if X.ndim == 2: # Primer, ko je vrnjen samo en spektrogram
        X = np.expand_dims(X, axis=0)
    return X, np.array(y)


def prepare_final_datasets(speaker_segments, train_ids, val_ids, test_ids):
    """
    Pripravi končne numpy arraye za učenje, vključno z one-hot kodiranjem.
    """
    print("\n--- 5. Priprava Podatkov za Model ---")
    all_unique_speakers = sorted(speaker_segments.keys())
    speaker_to_int = {speaker: i for i, speaker in enumerate(all_unique_speakers)}
    num_classes = len(all_unique_speakers)

    print("Ustvarjam učno množico...")
    X_train, y_train = create_dataset(train_ids, speaker_segments, speaker_to_int)
    print("Ustvarjam validacijsko množico...")
    X_val, y_val = create_dataset(val_ids, speaker_segments, speaker_to_int)
    print("Ustvarjam testno množico...")
    X_test, y_test = create_dataset(test_ids, speaker_segments, speaker_to_int)

    # Dodaj "kanalno" dimenzijo za CNN
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)

    # One-hot kodiranje oznak
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    print(f"Oblika učne množice X: {X_train.shape}, y: {y_train.shape}")
    print(f"Oblika validacijske množice X: {X_val.shape}, y: {y_val.shape}")
    print(f"Oblika testne množice X: {X_test.shape}, y: {y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes
