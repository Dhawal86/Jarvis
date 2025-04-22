import os
import random
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from collections import Counter

# Configuration
AUTHORIZED_DIR = "testing"
IMPOSTOR_DATASET_DIR = 'imposter_voices/mixvoice'
MODEL_PATH = "voice_ann_model.h5"
SAMPLE_RATE = 22050
DURATION = 3
AUTHORIZED_SAMPLES = 20
IMPOSTOR_SAMPLES = 20

os.makedirs(AUTHORIZED_DIR, exist_ok=True)

def extract_features(file_path, sample_rate=SAMPLE_RATE):
    try:
        if file_path.endswith(".npy"):
            audio = np.load(file_path).flatten()
        else:
            audio, _ = librosa.load(file_path, sr=sample_rate)

        target_len = sample_rate * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

        combined = np.hstack([np.mean(mfcc.T, axis=0),
                              np.mean(chroma.T, axis=0),
                              np.mean(contrast.T, axis=0)])
        return combined
    except Exception as e:
        print(f"Feature extraction failed for {file_path}: {e}")
        return None

def get_impostor_files():
    impostor_files = []
    for root, _, files in os.walk(IMPOSTOR_DATASET_DIR):
        for file in files:
            if file.lower().endswith('.wav'):
                impostor_files.append(os.path.join(root, file))

    print(f"Found {len(impostor_files)} impostor audio files.")
    return random.sample(impostor_files, min(IMPOSTOR_SAMPLES, len(impostor_files)))

def main():
    print("\nExtracting features from authorized samples...")
    X, y = [], []

    for file in os.listdir(AUTHORIZED_DIR):
        feature = extract_features(os.path.join(AUTHORIZED_DIR, file))
        if feature is not None:
            X.append(feature)
            y.append(1)

    print("\nExtracting features from impostor samples...")
    impostor_files = get_impostor_files()
    for file in impostor_files:
        feature = extract_features(file)
        if feature is not None:
            X.append(feature)
            y.append(0)

    if len(set(y)) < 2:
        print("Error: Not enough classes to train the model.")
        return

    print(f"\nClass distribution: {Counter(y)}")

    X = np.array(X)
    y = np.array(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Build ANN model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')  # 2 classes: authorized, impostor
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    print("\nTraining neural network...")
    model.fit(X_train, y_train_cat, epochs=30, batch_size=8, validation_split=0.2)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"\nTest Accuracy: {accuracy:.2f}")

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
