from collections import deque
import numpy as np
import cv2
import copy
import mediapipe as mp
import joblib

from ASL.utils import calc_landmark_list, pre_process_landmark

# Load trained model
model = joblib.load("asl_model.pkl")

# Store past 20 frames of finger tips
index_history = deque(maxlen=20)  # For Z

z_display_hold = 0
Z_DISPLAY_HOLD_FRAMES = 5

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def is_mostly_stationary(points, threshold=0.02):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) - min(xs) < threshold and max(ys) - min(ys) < threshold


def is_z_motion(points):
    if len(points) < 15:
        return False

    # Smooth out movement (rolling average)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Compute directional differences
    dx = np.diff(xs)
    dy = np.diff(ys)

    # Detect 3 clear horizontal direction changes (right → left → right)
    direction_changes = []
    for i in range(1, len(dx)):
        if dx[i - 1] * dx[i] < 0:  # sign change
            direction_changes.append(i)

    # Require at least 2 direction changes (for 3 segments)
    if len(direction_changes) < 2:
        return False

    # Ensure total horizontal movement is significant
    total_dx = np.sum(np.abs(dx))
    if total_dx < 0.3:  # You can tweak this value (e.g., 0.2 ~ 0.4)
        return False

    # Optional: make sure the movement is mostly horizontal (Z shape)
    vertical_range = max(ys) - min(ys)
    if vertical_range > 0.2:
        return False

    return True


# Cooldowns and trigger flags
cooldown = 0
z_static_ready = False
z_motion_timeout = 0  # countdown after Z is triggered by static gesture

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

            # Get pinky (20) and index (8) tips
            index_tip = hand_landmarks.landmark[8]

            index_history.append((index_tip.x, index_tip.y))

            # Predict static gesture
            prediction = model.predict([pre_processed_landmark_list])[0]
            final_output = prediction if prediction != "Z" else "Detecting Z motion..."

            if prediction == "Z":
                if not z_static_ready:
                    z_static_ready = True
                    z_motion_timeout = 40

            if z_static_ready and z_motion_timeout > 0:
                recent_points = list(index_history)
                if len(recent_points) == 20:
                    if is_z_motion(recent_points) and not is_mostly_stationary(recent_points[:5]):
                        final_output = "Z"
                        # index_history.clear()
                        cooldown = 30
                        z_static_ready = False
                        z_motion_timeout = 0
                        z_display_hold = Z_DISPLAY_HOLD_FRAMES
                    elif z_display_hold == 0:  # Only show detecting if not already confirmed Z
                        final_output = "Detecting Z motion..."

            if z_motion_timeout > 0:
                z_motion_timeout -= 1
            else:
                z_static_ready = False

            if z_display_hold > 0:
                final_output = "Z"
                z_display_hold -= 1

            # Display prediction
            cv2.putText(frame, f"Prediction: {final_output}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw index motion path (yellow)
            for i in range(1, len(index_history)):
                pt1 = index_history[i - 1]
                pt2 = index_history[i]
                cv2.line(frame,
                         (int(pt1[0] * frame.shape[1]), int(pt1[1] * frame.shape[0])),
                         (int(pt2[0] * frame.shape[1]), int(pt2[1] * frame.shape[0])),
                         (0, 255, 255), 2)

    # Decrease cooldowns
    if cooldown > 0:
        cooldown -= 1
    if z_motion_timeout > 0:
        z_motion_timeout -= 1

    # Show window
    cv2.imshow("ASL Real-Time Recognition - Press ESC to exit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
