import numpy as np
import pandas as pd
import os
import cv2
import mediapipe as mp
#MediaPipe Hand init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    max_detection_confidence=0.7,
)
#File Setup Directory
DATA_DIR= os.path.join(os.path.dirname(__file__),'..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'gestures.csv')

os.makedirs(DATA_DIR, exist_ok=True)

#VideoCapture Method

def collect_data():
    cap = cv2.VideoCapture(0)
    data=[]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Could not access camera")
            break
            #Frame Preprocessing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        #Landmark Extraction Logic
        current_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                current_landmarks = landmark_list
        #UI & Keyboard Input
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
    #Exporting to CSV
    cap.release()
    cv2.destroyAllWindows()

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



