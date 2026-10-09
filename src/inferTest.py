import os
import time
from collections import deque

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Path
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_NAMES_PATH = os.path.join(MODEL_DIR, "label_names.joblib")
MODEL_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

#Load Trained Models
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
LABEL_NAMES = joblib.load(LABEL_NAMES_PATH)
print(f"Loaded model. Classes: {LABEL_NAMES}")

#MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandResultHolder:
    def __init__(self):
        self.result = None

    def callback(self, result, output_image, timestamp_ms):
        self.result = result


#Normalization
def normalize_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    pts = flat_landmarks.reshape(21, 3).astype(np.float64)
    pts = pts - pts[0]
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts.flatten()


#Drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks_on_frame(frame, hand_landmarks):
    h, w, _ = frame.shape
    for a, b in HAND_CONNECTIONS:
        p1, p2 = hand_landmarks[a], hand_landmarks[b]
        cv2.line(frame,
                 (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)),
                 (0, 255, 0), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)


#Main
def run():
    cap = cv2.VideoCapture(0)
    holder = HandResultHolder()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=holder.callback,
    )

    # Rolling window for smoothing predictions
    history = deque(maxlen=10)

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, ts_ms)

            result = holder.result
            label_text = "no hand"
            confidence_text = ""

            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks_on_frame(frame, hand_landmarks)

                    # Build feature vector
                    flat = []
                    for lm in hand_landmarks:
                        flat.extend([lm.x, lm.y, lm.z])
                    flat = np.array(flat)

                    # Same normalization as training
                    flat_norm = normalize_landmarks(flat).reshape(1, -1)

                    # Same scaling as training
                    flat_scaled = scaler.transform(flat_norm)

                    # Predict
                    pred = clf.predict(flat_scaled)[0]
                    probs = clf.predict_proba(flat_scaled)[0]
                    conf = probs.max()

                    history.append(int(pred))

                    # Majority vote over last N frames
                    from collections import Counter
                    smoothed = Counter(history).most_common(1)[0][0]

                    name = LABEL_NAMES.get(smoothed, f"class_{smoothed}")
                    label_text = f"{name}"
                    confidence_text = f"conf: {conf:.2f}"
            else:
                history.clear()

            # Draw overlay
            cv2.putText(frame, f"Prediction: {label_text}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if confidence_text:
                cv2.putText(frame, confidence_text,
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Gesture Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()