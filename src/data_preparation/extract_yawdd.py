import cv2, os, glob

RAW_YAWDD = "data/raw/yawdd"
OUT_PATH = "data/raw/yawdd_frames"
os.makedirs(OUT_PATH, exist_ok=True)

for vid in glob.glob(os.path.join(RAW_YAWDD, "*.avi")):
    cap = cv2.VideoCapture(vid)
    name = os.path.splitext(os.path.basename(vid))[0]
    frame_dir = os.path.join(OUT_PATH, name)
    os.makedirs(frame_dir, exist_ok=True)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save every 10th frame to reduce data size
        if count % 10 == 0:
            cv2.imwrite(os.path.join(frame_dir, f"{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"✅ Extracted {count} frames from {vid} → {frame_dir}")
