import os
import random
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# Configuration
AUTHORIZED_DIR = 'testing'
IMPOSTOR_DATASET_DIR = 'imposter_voices/mixvoice'
MODEL_PATH = "voice_model_tree.pkl"
SAMPLE_RATE = 22050
DURATION = 3
AUTHORIZED_SAMPLES = 20
IMPOSTOR_SAMPLES = 20
ERROR_LOG = "feature_errors.log"
CRITERION = "gini"  # Use "gini" for CART, or "entropy" for ID3

def extract_features_from_array(audio, sample_rate=SAMPLE_RATE):
    try:
        target_length = sample_rate * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

        return np.hstack([mfccs, chroma, zcr])

    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return None

def extract_features(file_path, sample_rate=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        return extract_features_from_array(audio, sample_rate)
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
    # Feature extraction
    print("\nExtracting authorized user features...")
    X, y = [], []
    authorized_files = [f for f in os.listdir(AUTHORIZED_DIR) if f.lower().endswith(".wav")]
    if len(authorized_files) < AUTHORIZED_SAMPLES:
        print(f"[Warning] Found only {len(authorized_files)} authorized samples. Proceeding with available files.")
    for file in authorized_files[:AUTHORIZED_SAMPLES]:
        path = os.path.join(AUTHORIZED_DIR, file)
        feature = extract_features(path)
        if feature is not None:
            X.append(feature)
            y.append(1)

    print("\nExtracting impostor features...")
    impostor_files = get_impostor_files()
    if len(impostor_files) < IMPOSTOR_SAMPLES:
        print(f"[Warning] Only {len(impostor_files)} impostor samples found. Adjusting count.")

    for file_path in impostor_files:
        feature = extract_features(file_path)
        if feature is not None:
            X.append(feature)
            y.append(0)

    if len(set(y)) < 2:
        print("\nError: Not enough classes to train. Need both authorized and impostor samples.")
        return

    print("\nClass distribution:", Counter(y))

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Decision Tree classifier...")
    model = DecisionTreeClassifier(criterion=CRITERION, max_depth=None, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)

    print(f"\nModel trained successfully with accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump((model, scaler), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Visualization
    feature_names = [f"mfcc_{i}" for i in range(13)] + [f"chroma_{i}" for i in range(12)] + ["zcr"]
    plt.figure(figsize=(16, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=["Impostor", "Authorized"])
    plt.title("Decision Tree Visualization")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Impostor", "Authorized"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
