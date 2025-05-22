from collections import deque
import numpy as np
import cv2
import copy
import mediapipe as mp
import joblib
import copy
import itertools

import tkinter as tk
from tkinter import messagebox, ttk
import threading
from PIL import Image, ImageTk


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_width - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# Load trained model
try:
    model = joblib.load("asl_model.pkl")
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

# Store past 20 frames of finger tips
index_history = deque(maxlen=20)  # For Z

z_display_hold = 0
Z_DISPLAY_HOLD_FRAMES = 5

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Cooldowns and trigger flags
cooldown = 0
z_static_ready = False
z_motion_timeout = 0  # countdown after Z is triggered by static gesture


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


def start_inference():
    global cooldown, z_static_ready, z_motion_timeout, z_display_hold
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


class ModernASLApp:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()

    def setup_window(self):
        self.root.title("ASL Recognition System")
        self.root.geometry("600x700")
        self.root.resizable(False, False)

        # Configure colors
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'accent': '#E74C3C',
            'success': '#27AE60',
            'background': '#ECF0F1',
            'card': '#FFFFFF',
            'text': '#2C3E50',
            'text_light': '#7F8C8D'
        }

        self.root.configure(bg=self.colors['background'])

        # Configure styles
        self.setup_styles()

    def setup_styles(self):
        style = ttk.Style()

        # Configure button styles
        style.configure('Primary.TButton',
                        font=('Segoe UI', 12, 'bold'),
                        padding=(20, 15))

        style.configure('Secondary.TButton',
                        font=('Segoe UI', 10),
                        padding=(15, 10))

        style.configure('Status.TLabel',
                        font=('Segoe UI', 10),
                        foreground=self.colors['text_light'])

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # Header section
        self.create_header(main_frame)

        # Status card
        self.create_status_card(main_frame)

        # Main action card
        self.create_action_card(main_frame)

        # Information card
        self.create_info_card(main_frame)

        # Footer
        self.create_footer(main_frame)

    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill='x', pady=(0, 30))

        # App icon/logo placeholder
        icon_frame = tk.Frame(header_frame,
                              bg=self.colors['secondary'],
                              width=60, height=60)
        icon_frame.pack_propagate(False)
        icon_frame.pack(side='left')

        icon_label = tk.Label(icon_frame,
                              text="✋",
                              font=('Segoe UI', 24),
                              bg=self.colors['secondary'],
                              fg='white')
        icon_label.pack(expand=True)

        # Title and subtitle
        title_frame = tk.Frame(header_frame, bg=self.colors['background'])
        title_frame.pack(side='left', padx=(20, 0), fill='x', expand=True)

        title_label = tk.Label(title_frame,
                               text="ASL Recognition System",
                               font=('Segoe UI', 24, 'bold'),
                               bg=self.colors['background'],
                               fg=self.colors['primary'])
        title_label.pack(anchor='w')

        subtitle_label = tk.Label(title_frame,
                                  text="Real-time American Sign Language Detection",
                                  font=('Segoe UI', 12),
                                  bg=self.colors['background'],
                                  fg=self.colors['text_light'])
        subtitle_label.pack(anchor='w')

    def create_status_card(self, parent):
        status_card = tk.Frame(parent,
                               bg=self.colors['card'],
                               relief='flat',
                               bd=1)
        status_card.pack(fill='x', pady=(0, 20))

        # Add subtle shadow effect
        shadow = tk.Frame(parent, bg='#BDC3C7', height=2)
        shadow.pack(fill='x', pady=(0, 18))

        status_inner = tk.Frame(status_card, bg=self.colors['card'])
        status_inner.pack(fill='both', expand=True, padx=25, pady=20)

        status_title = tk.Label(status_inner,
                                text="System Status",
                                font=('Segoe UI', 14, 'bold'),
                                bg=self.colors['card'],
                                fg=self.colors['primary'])
        status_title.pack(anchor='w')

        # Model status
        model_status_frame = tk.Frame(status_inner, bg=self.colors['card'])
        model_status_frame.pack(fill='x', pady=(10, 5))

        status_color = self.colors['success'] if MODEL_LOADED else self.colors['accent']
        status_text = "✓ Model Loaded Successfully" if MODEL_LOADED else "✗ Model Not Found"

        status_indicator = tk.Label(model_status_frame,
                                    text="●",
                                    font=('Segoe UI', 16),
                                    fg=status_color,
                                    bg=self.colors['card'])
        status_indicator.pack(side='left')

        status_label = tk.Label(model_status_frame,
                                text=status_text,
                                font=('Segoe UI', 11),
                                bg=self.colors['card'],
                                fg=self.colors['text'])
        status_label.pack(side='left', padx=(10, 0))

        # Camera status
        camera_frame = tk.Frame(status_inner, bg=self.colors['card'])
        camera_frame.pack(fill='x', pady=5)

        camera_indicator = tk.Label(camera_frame,
                                    text="●",
                                    font=('Segoe UI', 16),
                                    fg=self.colors['text_light'],
                                    bg=self.colors['card'])
        camera_indicator.pack(side='left')

        camera_label = tk.Label(camera_frame,
                                text="Camera Ready",
                                font=('Segoe UI', 11),
                                bg=self.colors['card'],
                                fg=self.colors['text'])
        camera_label.pack(side='left', padx=(10, 0))

    def create_action_card(self, parent):
        action_card = tk.Frame(parent,
                               bg=self.colors['card'],
                               relief='flat',
                               bd=1)
        action_card.pack(fill='x', pady=20)

        # Shadow
        shadow = tk.Frame(parent, bg='#BDC3C7', height=2)
        shadow.pack(fill='x', pady=(0, 18))

        action_inner = tk.Frame(action_card, bg=self.colors['card'])
        action_inner.pack(fill='both', expand=True, padx=25, pady=30)

        # Start button
        start_button = tk.Button(action_inner,
                                 text="🎥 Start Recognition",
                                 font=('Segoe UI', 16, 'bold'),
                                 bg=self.colors['secondary'],
                                 fg='white',
                                 relief='flat',
                                 bd=0,
                                 padx=40,
                                 pady=15,
                                 cursor='hand2',
                                 command=self.start_recognition)
        start_button.pack(pady=(0, 15))

        # Add hover effect
        def on_enter(e):
            start_button.configure(bg='#2980B9')

        def on_leave(e):
            start_button.configure(bg=self.colors['secondary'])

        start_button.bind("<Enter>", on_enter)
        start_button.bind("<Leave>", on_leave)

        # Instructions button
        instructions_button = tk.Button(action_inner,
                                        text="📋 How to Use",
                                        font=('Segoe UI', 12),
                                        bg=self.colors['background'],
                                        fg=self.colors['text'],
                                        relief='flat',
                                        bd=1,
                                        padx=30,
                                        pady=10,
                                        cursor='hand2',
                                        command=self.show_instructions)
        instructions_button.pack()

    def create_info_card(self, parent):
        info_card = tk.Frame(parent,
                             bg=self.colors['card'],
                             relief='flat',
                             bd=1)
        info_card.pack(fill='both', expand=True, pady=20)

        # Shadow
        shadow = tk.Frame(parent, bg='#BDC3C7', height=2)
        shadow.pack(fill='x', pady=(0, 18))

        info_inner = tk.Frame(info_card, bg=self.colors['card'])
        info_inner.pack(fill='both', expand=True, padx=25, pady=20)

        info_title = tk.Label(info_inner,
                              text="Quick Guide",
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.colors['card'],
                              fg=self.colors['primary'])
        info_title.pack(anchor='w', pady=(0, 15))

        # Guide steps
        steps = [
            "1. Ensure good lighting and clear background",
            "2. Position your hand clearly in front of the camera",
            "3. Make distinct ASL gestures",
            "4. For letter 'Z', draw the shape with your index finger",
            "5. Press ESC to exit the recognition window"
        ]

        for step in steps:
            step_label = tk.Label(info_inner,
                                  text=step,
                                  font=('Segoe UI', 10),
                                  bg=self.colors['card'],
                                  fg=self.colors['text'],
                                  justify='left')
            step_label.pack(anchor='w', pady=2)

    def create_footer(self, parent):
        footer_frame = tk.Frame(parent, bg=self.colors['background'])
        footer_frame.pack(side='bottom', fill='x', pady=(20, 0))

        footer_label = tk.Label(footer_frame,
                                text="Powered by MediaPipe & OpenCV",
                                font=('Segoe UI', 9),
                                bg=self.colors['background'],
                                fg=self.colors['text_light'])
        footer_label.pack(side='right')

    def start_recognition(self):
        if not MODEL_LOADED:
            messagebox.showerror("Error",
                                 "Model file 'asl_model.pkl' not found!\nPlease ensure the model file is in the same directory.")
            return

        self.root.withdraw()  # Hide the main window
        threading.Thread(target=self.run_cv_loop, daemon=True).start()

    def run_cv_loop(self):
        try:
            start_inference()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.root.destroy()

    def show_instructions(self):
        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("How to Use - ASL Recognition")
        instructions_window.geometry("500x600")
        instructions_window.configure(bg=self.colors['background'])
        instructions_window.resizable(False, False)

        # Make it modal
        instructions_window.transient(self.root)
        instructions_window.grab_set()

        # Center the window
        instructions_window.geometry("+{}+{}".format(
            int(instructions_window.winfo_screenwidth() / 2 - 250),
            int(instructions_window.winfo_screenheight() / 2 - 300)
        ))

        main_frame = tk.Frame(instructions_window, bg=self.colors['background'])
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # Title
        title = tk.Label(main_frame,
                         text="How to Use ASL Recognition",
                         font=('Segoe UI', 18, 'bold'),
                         bg=self.colors['background'],
                         fg=self.colors['primary'])
        title.pack(pady=(0, 20))

        # Instructions text
        instructions_text = """
Getting Started:
• Click 'Start Recognition' to launch the camera interface
• Position yourself 2-3 feet from the camera
• Ensure good lighting and minimal background clutter

Making Gestures:
• Hold your hand steady and clearly visible
• Make distinct ASL letter signs
• Wait for the system to recognize each gesture
• The prediction will appear on screen in real-time

Special Instructions for 'Z':
• First, make the static 'Z' hand sign
• When prompted, trace the letter 'Z' in the air with your index finger
• Draw from top-left to top-right, then diagonal down, then bottom-left to bottom-right

Tips for Best Results:
• Use consistent hand positioning
• Avoid rapid movements between gestures
• Practice gestures beforehand for accuracy
• Ensure your entire hand is visible in the frame

Troubleshooting:
• If recognition is poor, adjust lighting
• Make sure your hand contrasts with the background
• Try slightly different hand positions
• Press ESC to exit and restart if needed
        """

        text_widget = tk.Text(main_frame,
                              wrap='word',
                              font=('Segoe UI', 10),
                              bg=self.colors['card'],
                              fg=self.colors['text'],
                              relief='flat',
                              bd=0,
                              padx=20,
                              pady=20)
        text_widget.pack(fill='both', expand=True)
        text_widget.insert('1.0', instructions_text.strip())
        text_widget.configure(state='disabled')

        # Close button
        close_button = tk.Button(main_frame,
                                 text="Got it!",
                                 font=('Segoe UI', 12, 'bold'),
                                 bg=self.colors['secondary'],
                                 fg='white',
                                 relief='flat',
                                 bd=0,
                                 padx=30,
                                 pady=10,
                                 cursor='hand2',
                                 command=instructions_window.destroy)
        close_button.pack(pady=(20, 0))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ModernASLApp()
    app.run()