import os
import cv2
import copy
import mediapipe as mp

from ASL.utils import calc_landmark_list, pre_process_landmark

label = input("Enter the label (A-Z) for this data collection: ")

data_file = "asl_data.csv"

data_count = 0
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        for line in f:
            if line.strip().endswith(f",{label}"):
                data_count += 1

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    debug_image = copy.deepcopy(frame)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Save to CSV
            with open(data_file, "a") as f:
                f.write(",".join(map(str, pre_processed_landmark_list)) + f",{label}\n")
            data_count += 1

    cv2.putText(frame, f"Label: {label} | Data points: {data_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Data Collection - Press ESC to stop", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
