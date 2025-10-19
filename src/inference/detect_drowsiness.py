import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

VIDEO_PATH = "sample_cabin_video.mp4"  
EYE_MODEL_PATH = "models/eye_state_cnn.h5"
YAWN_MODEL_PATH = "models/yawn_cnn.h5"

EYE_IMG_SIZE = (64, 64)
MOUTH_IMG_SIZE = (64, 64)
EYE_CLOSED_THRESH = 0.5
YAWN_THRESH = 0.5
SMOOTH_WINDOW = 5
DEBUG_MODE = True  


print("🔹 Loading trained models...")
eye_model = load_model(EYE_MODEL_PATH, compile=False)
yawn_model = load_model(YAWN_MODEL_PATH, compile=False)
print("✅ Models loaded successfully!")

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDXS  = [33, 133, 159, 145, 153, 154, 155, 173, 157, 158, 160, 161, 246]
RIGHT_EYE_IDXS = [362, 263, 386, 374, 380, 381, 382, 390, 384, 385, 387, 388, 466]
LIPS_IDXS      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318]

def extract_roi(frame, landmarks, indices, pad=0.3):
    #extracting roi information
    h, w = frame.shape[:2]
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x0, x1 = int(max(0, min(xs))), int(min(w, max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(h, max(ys)))
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, x1 = max(0, x0 - dx), min(w, x1 + dx)
    y0, y1 = max(0, y0 - dy), min(h, y1 + dy)
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0, 0, 0)
    roi = frame[y0:y1, x0:x1]
    return roi if roi.size else None, (x0, y0, x1, y1)

if os.path.exists(VIDEO_PATH):
    print(f"Using video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    print("No video file found. Opening webcam.")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open video source.")
    raise SystemExit

print("Starting real-time drowsiness detection. Press 'q' to quit.")
drowsy_start = None
alert_on = False
eye_q, yawn_q = deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)

show_rois = False

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                           refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or camera not available.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eye_closed_prob, yawn_prob = 0.0, 0.0
        both_rgb, m_rgb = None, None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left, box_l = extract_roi(frame, lm, LEFT_EYE_IDXS)
            right, box_r = extract_roi(frame, lm, RIGHT_EYE_IDXS)
            mouth, box_m = extract_roi(frame, lm, LIPS_IDXS)

            #eyes
            if left is not None and right is not None:
                try:
                    # Ensure 3 channels
                    if left.ndim == 2:
                        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
                    if right.ndim == 2:
                        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

                    # Resize both individually to 64x64
                    left = cv2.resize(left, EYE_IMG_SIZE)
                    right = cv2.resize(right, EYE_IMG_SIZE)

                    # Convert both to RGB, normalize, and predict separately
                    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
                    left_input = np.expand_dims(left_rgb / 255.0, 0).astype(np.float32)
                    right_input = np.expand_dims(right_rgb / 255.0, 0).astype(np.float32)

                    left_open = float(eye_model.predict(left_input, verbose=0)[0][0])
                    right_open = float(eye_model.predict(right_input, verbose=0)[0][0])

                    # Average both sides
                    eye_open_prob = (left_open + right_open) / 2.0
                    eye_closed_prob = 1.0 - eye_open_prob
                    eye_q.append(eye_closed_prob)

                    if DEBUG_MODE:
                        print(f"Eye ROI mean pixel (L,R): {np.mean(left_rgb):.1f}, {np.mean(right_rgb):.1f}")

                except Exception as e:
                    if DEBUG_MODE:
                        print("Eye predict error:", e)
            else:
                if DEBUG_MODE:
                    print("One or both eye ROIs missing")

            #mouth
            if mouth is not None:
                try:
                    m_rgb = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
                    m_rgb = cv2.resize(m_rgb, MOUTH_IMG_SIZE)
                    mouth_input = np.expand_dims(m_rgb / 255.0, 0).astype(np.float32)

                    yawn_prob = float(yawn_model.predict(mouth_input, verbose=0)[0][0])
                    yawn_q.append(yawn_prob)
                except Exception as e:
                    if DEBUG_MODE:
                        print("Yawn predict err:", e)

        # Smoothed probabilities
        eye_closed_s = np.mean(eye_q) if eye_q else 0.0
        yawn_s = np.mean(yawn_q) if yawn_q else 0.0
        eyes_closed = eye_closed_s > EYE_CLOSED_THRESH
        yawn_yes = yawn_s > YAWN_THRESH

        # Debug print
        if DEBUG_MODE:
            print(f"EyeClosed={eye_closed_s:.2f} | Yawn={yawn_s:.2f}")

        # Drowsiness logic
        if eyes_closed and yawn_yes:
            if not drowsy_start:
                drowsy_start = time.time()
            elif time.time() - drowsy_start > 2.0:
                alert_on = True
        else:
            drowsy_start = None
            alert_on = False

        #UI
        cv2.putText(frame, f"Eyes: {'Closed' if eyes_closed else 'Open'} ({eye_closed_s:.2f})",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Yawn: {'Yes' if yawn_yes else 'No'} ({yawn_s:.2f})",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if alert_on:
            cv2.putText(frame, "⚠️ DROWSINESS ALERT ⚠️", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited cleanly.")
