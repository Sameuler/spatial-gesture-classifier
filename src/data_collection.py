import numpy as np
import pandas as pd
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#MediaPipe Hand init
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


#File Setup Directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

latest_result = {"result": None}


def _on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback for LIVE_STREAM mode."""
    latest_result["result"] = result

DATA_DIR= os.path.join(os.path.dirname(__file__),'..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'gestures.csv')
os.makedirs(DATA_DIR, exist_ok=True)
#Manual Drawing Helper
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                   # palm
]

#HandLandmark Drawing
def draw_landmarks_on_frame(frame, hand_landmarks):
    h, w, _ = frame.shape
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = hand_landmarks[start_idx]
        p2 = hand_landmarks[end_idx]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (0, 255, 0), 2,
        )
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

#VideoCapture Method

def collect_data():
    cap = cv2.VideoCapture(0)
    data = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=_on_result,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Could not access camera")
                break
                # Frame Preprocessing
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Landmark Extraction Logic
            current_landmarks = None
            result = latest_result["result"]

            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks_on_frame(frame, hand_landmarks)

                    landmark_list = []
                    for lm in hand_landmarks:
                        landmark_list.extend([lm.x, lm.y, lm.z])

                    current_landmarks = landmark_list
            # UI & Keyboard Input
            cv2.putText(
                frame, "Press 0-9 to log gesture | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("Spatial Gesture Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                label = chr(key)
                if current_landmarks is not None:
                    # Append landmark vector + label
                    row = current_landmarks + [int(label)]
                    data.append(row)
                    print(f"Recorded sample for Label [{label}] | Total samples: {len(data)}")
                else:
                    print(f"No hand detected! Position hand in view to log Label [{label}].")
    cap.release()
    cv2.destroyAllWindows()
    #Exporting to CSV
    if data:
        columns = [f"lm_{i}" for i in range(63)] + ["label"]
        df = pd.DataFrame(data, columns=columns)

        if os.path.exists(CSV_PATH):
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_PATH, index=False)

        print(f"\nSuccessfully saved {len(df)} samples to {CSV_PATH}")
if __name__ == '__main__':
    collect_data()



