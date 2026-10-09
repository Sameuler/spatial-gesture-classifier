import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

#Gestures Labels
LABEL_NAMES = {
    0: "open_palm",
    1: "index",
    2: "peace",
    3: "ok_sign",
    4: "thumbs_up",
    5: "fist",
}

RANDOM_STATE = 42 #seed
TEST_SIZE = 0.2 #20%testing 80%training

#Normalization

def normalize_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    pts = flat_landmarks.reshape(21, 3).astype(np.float64)
    pts = pts - pts[0]  # wrist-relative
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts.flatten()
def normalize_dataset(X: np.ndarray) -> np.ndarray:
    return np.stack([normalize_landmarks(row) for row in X])

#Load

def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No dataset found at {CSV_PATH}. Collect sample using the collection script."
        )

    df = pd.read_csv(CSV_PATH)
    expected_cols = [f"lm_{i}" for i in range(63)] + ["label"]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Unexpected columns. Expected {expected_cols[:3]}...{expected_cols[-1]}, "
            f"got {list(df.columns)[:3]}...{list(df.columns)[-1]}"
        )

    X = df[[f"lm_{i}" for i in range(63)]].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    print(f"Loaded {len(df)} samples, {X.shape[1]} features per sample.")
    print("Class distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        name = LABEL_NAMES.get(int(label), f"class_{label}")
        print(f"  [{label}] {name:<12} {count} samples")

    return X, y


#Training

X_raw, y = load_data()

#Normalize

print('Normalizing data...')
X_norm = normalize_dataset(X_raw)

#Train
X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

#Standardize (mean=0, standard deviation=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Classifier

print("Training Random Forest")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
clf.fit(X_train_scaled, y_train)

#Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {acc:.4f}\n")
print("Classification report:")
target_names = [LABEL_NAMES.get(int(l), f"class_{l}")
                for l in np.unique(y)]
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Save

joblib.dump(clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(LABEL_NAMES, os.path.join(MODEL_DIR, "label_names.joblib"))

print(f"Saved model     -> {MODEL_PATH}")
print(f"Saved scaler    -> {SCALER_PATH}")
print(f"Saved labels    -> {os.path.join(MODEL_DIR, 'label_names.joblib')}")


if __name__ == "__main__":
    train()




