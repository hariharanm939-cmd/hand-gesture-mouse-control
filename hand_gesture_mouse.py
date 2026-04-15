import time
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import pyautogui
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    drawing_utils,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# Download model if not present
MODEL_PATH = Path("hand_landmarker.task")
if not MODEL_PATH.exists():
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")

# Initialize hand landmarker in VIDEO mode for better performance
options = HandLandmarkerOptions(
    base_options={"model_asset_path": str(MODEL_PATH)},
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
hand_landmarker = HandLandmarker.create_from_options(options)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Reduced smoothing for faster response
smoothening = 2
prev_x, prev_y = 0, 0

last_click_time = 0.0
click_delay = 0.3

last_volume_time = 0.0
volume_delay = 0.3

last_mute_time = 0.0
mute_delay = 0.3

pyautogui.FAILSAFE = False

def adjust_screen_coords(value, screen_dim, scaling_factor=1.5):
    return int(value * screen_dim * scaling_factor)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(ImageFormat.SRGB, rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )

            index_finger = hand_landmarks[8]
            thumb_finger = hand_landmarks[4]
            middle_finger = hand_landmarks[12]
            ring_finger = hand_landmarks[16]
            pinky_finger = hand_landmarks[20]

            x, y = int(index_finger.x * w), int(index_finger.y * h)
            screen_x = adjust_screen_coords(index_finger.x, screen_w)
            screen_y = adjust_screen_coords(index_finger.y, screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            thumb_x, thumb_y = int(thumb_finger.x * w), int(thumb_finger.y * h)
            middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
            ring_x, ring_y = int(ring_finger.x * w), int(ring_finger.y * h)
            pinky_x, pinky_y = int(pinky_finger.x * w), int(pinky_finger.y * h)
            now = time.time()

            if abs(thumb_x - x) < 20 and abs(thumb_y - y) < 20:
                if now - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = now
                cv2.putText(frame, "Click", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if abs(middle_x - thumb_x) < 20 and abs(middle_y - thumb_y) < 20:
                if now - last_volume_time > volume_delay:
                    pyautogui.press("volumeup")
                    last_volume_time = now
                cv2.putText(frame, "Volume Up", (middle_x, middle_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if abs(ring_x - thumb_x) < 20 and abs(ring_y - thumb_y) < 20:
                if now - last_volume_time > volume_delay:
                    pyautogui.press("volumedown")
                    last_volume_time = now
                cv2.putText(frame, "Volume Down", (ring_x, ring_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if abs(pinky_x - thumb_x) < 20 and abs(pinky_y - thumb_y) < 20:
                if now - last_mute_time > mute_delay:
                    pyautogui.press("volumemute")
                    last_mute_time = now
                cv2.putText(frame, "Mute", (pinky_x, pinky_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
