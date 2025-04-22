import os
import random
import numpy as np
import librosa
import sounddevice as sd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from collections import Counter

# Configuration
AUTHORIZED_DIR = "voice_samples/authorized"
IMPOSTOR_DATASET_DIR = 'imposter_voices/mixvoice'
MODEL_PATH = "voice_model.pkl"
SAMPLE_RATE = 22050
DURATION = 3
AUTHORIZED_SAMPLES = 20
IMPOSTOR_SAMPLES = 20
ERROR_LOG = "feature_errors.log"

# Ensure authorized directory exists
os.makedirs(AUTHORIZED_DIR, exist_ok=True)

def record_voice(save_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    print(f"Recording {save_path}... Speak now!")
    recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()
    np.save(save_path, recording)
    print(f"Saved {save_path}")

# Optional audio augmentation (disabled by default)
def augment_audio(audio):
    noise = np.random.randn(len(audio))
    return audio + 0.005 * noise

def extract_features(file_path, sample_rate=SAMPLE_RATE):
    try:
        if file_path.endswith(".npy"):
            audio = np.load(file_path).flatten()
        else:
            audio, _ = librosa.load(file_path, sr=sample_rate)

        # Pad/trim
        target_length = sample_rate * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # Optional: apply augmentation
        # audio = augment_audio(audio)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)

    except Exception as e:
        with open(ERROR_LOG, "a") as f:
            f.write(f"{file_path}: {e}\n")
        print(f"[Error] Skipped {file_path}")
        return None

def get_impostor_files():
    all_audio_files = []

    for root, dirs, files in os.walk(IMPOSTOR_DATASET_DIR):
        for file in files:
            if file.lower().endswith(".wav"):
                all_audio_files.append(os.path.join(root, file))

    print(f"Found {len(all_audio_files)} impostor audio files.")
    if not all_audio_files:
        return []

    sample_count = min(IMPOSTOR_SAMPLES, len(all_audio_files))
    return random.sample(all_audio_files, sample_count)

def main():
    # Prompt if authorized samples already exist
    if os.listdir(AUTHORIZED_DIR):
        choice = input("Authorized samples already exist. Re-record? (y/n): ").strip().lower()
        if choice == 'y':
            for i in range(AUTHORIZED_SAMPLES):
                filename = os.path.join(AUTHORIZED_DIR, f"auth_sample_{i}.npy")
                record_voice(filename)
        else:
            print("Using existing authorized samples.")
    else:
        print("Recording authorized user samples...")
        for i in range(AUTHORIZED_SAMPLES):
            filename = os.path.join(AUTHORIZED_DIR, f"auth_sample_{i}.npy")
            record_voice(filename)

    # Feature extraction
    print("\nExtracting authorized user features...")
    X, y = [], []
    for file in os.listdir(AUTHORIZED_DIR):
        feature = extract_features(os.path.join(AUTHORIZED_DIR, file))
        if feature is not None:
            X.append(feature)
            y.append(1)

    print("\nExtracting impostor features...")
    impostor_files = get_impostor_files()
    if len(impostor_files) < IMPOSTOR_SAMPLES:
        print(f" Only {len(impostor_files)} impostor samples found. Adjusting count.")

    for file_path in impostor_files:
        feature = extract_features(file_path)
        if feature is not None:
            X.append(feature)
            y.append(0)

    if len(set(y)) < 2:
        print("\n Error: Not enough classes to train. Need both authorized and impostor samples.")
        return

    print("\nClass distribution:", Counter(y))

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)

    print(f"\n Model trained successfully with accuracy: {accuracy:.2f}")
    print("\n Classification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump((model, scaler), MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
