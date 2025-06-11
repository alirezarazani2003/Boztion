import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

from utils.feature_extractor import extract_features

DATASET_PATH = "data/en"

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_emotion_from_filename(filename):
    parts = filename.split('-')
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, 'unknown')

def main():
    features = []
    labels = []

    print("Extracting features from audio files...")
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = parse_emotion_from_filename(file)
                if emotion == 'unknown':
                    continue
                feature = extract_features(file_path)
                features.append(feature)
                labels.append(emotion)

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training boztion model (RandomForest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    with open("model/boztion.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training complete and saved as model/boztion.pkl")

if __name__ == "__main__":
    main()
