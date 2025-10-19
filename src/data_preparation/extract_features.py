import os
import time
from tqdm import tqdm
import cv2
import mediapipe as mp

RAW_DIR = "data/cleaned"          # contains: eyes_open, eyes_closed, yawn, no_yawn
PROC_DIR = "data/processed"       # we'll save into class subfolders here

# Make class subfolders we actually train on
EYES_OPEN_DIR   = os.path.join(PROC_DIR, "eyes", "open")
EYES_CLOSED_DIR = os.path.join(PROC_DIR, "eyes", "closed")
MOUTH_YAWN_DIR  = os.path.join(PROC_DIR, "mouth", "yawn")
MOUTH_NO_YAWN_DIR = os.path.join(PROC_DIR, "mouth", "no_yawn")
for d in [EYES_OPEN_DIR, EYES_CLOSED_DIR, MOUTH_YAWN_DIR, MOUTH_NO_YAWN_DIR]:
    os.makedirs(d, exist_ok=True)

#Crop sizes
EYE_IMG_SIZE = (64, 64)
MOUTH_IMG_SIZE = (64, 64)

mp_face_mesh = mp.solutions.face_mesh

# The SAME landmark indices used in inference
LEFT_EYE_IDXS  = [33, 133, 159, 145, 153, 154, 155, 173, 157, 158, 160, 161, 246]
RIGHT_EYE_IDXS = [362, 263, 386, 374, 380, 381, 382, 390, 384, 385, 387, 388, 466]
LIPS_IDXS      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318]

def extract_roi(frame, landmarks, indices, pad=0.25):
    #using roi
    h, w = frame.shape[:2]
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x0, x1 = int(max(0, min(xs))), int(min(w, max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(h, max(ys)))
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, x1 = max(0, x0 - dx), min(w, x1 + dx)
    y0, y1 = max(0, y0 - dy), min(h, y1 + dy)
    roi = frame[y0:y1, x0:x1]
    return roi if roi.size else None

def process_folder(folder_name: str):
    #eyes open, close
    #mouth open, close
    folder_path = os.path.join(RAW_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return

    mode = None
    if folder_name == "eyes_open":
        mode = "eyes_open"
    elif folder_name == "eyes_closed":
        mode = "eyes_closed"
    elif folder_name == "yawn":
        mode = "yawn"
    elif folder_name == "no_yawn":
        mode = "no_yawn"
    else:
        return

    start_time = time.time()
    skipped, saved = 0, 0

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
    if not files:
        print(f" No images in {folder_name}")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for f in tqdm(files, desc=f"Processing {folder_name}", ncols=100):
            p = os.path.join(folder_path, f)
            frame = cv2.imread(p)
            if frame is None:
                skipped += 1
                continue

            # Face landmarks
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                skipped += 1
                continue

            lm = results.multi_face_landmarks[0].landmark

            # EYES path
            if mode in ("eyes_open", "eyes_closed"):
                left = extract_roi(frame, lm, LEFT_EYE_IDXS)
                right = extract_roi(frame, lm, RIGHT_EYE_IDXS)
                if left is None or right is None:
                    skipped += 1
                    continue
                try:
                    left  = cv2.resize(left,  EYE_IMG_SIZE)
                    right = cv2.resize(right, EYE_IMG_SIZE)
                    both  = cv2.hconcat([left, right])
                except Exception:
                    skipped += 1
                    continue

                out_dir = EYES_OPEN_DIR if mode == "eyes_open" else EYES_CLOSED_DIR
                cv2.imwrite(os.path.join(out_dir, f), both)
                saved += 1

            # MOUTH path
            if mode in ("yawn", "no_yawn"):
                mouth = extract_roi(frame, lm, LIPS_IDXS)
                if mouth is None:
                    skipped += 1
                    continue
                try:
                    mouth = cv2.resize(mouth, MOUTH_IMG_SIZE)
                except Exception:
                    skipped += 1
                    continue
                out_dir = MOUTH_YAWN_DIR if mode == "yawn" else MOUTH_NO_YAWN_DIR
                cv2.imwrite(os.path.join(out_dir, f), mouth)
                saved += 1

    dt = time.time() - start_time
    print(f"✅ {folder_name}: Saved {saved}, Skipped {skipped}, Time: {dt:.1f}s")

def main():
    for folder in os.listdir(RAW_DIR):
        process_folder(folder)

if __name__ == "__main__":
    main()
