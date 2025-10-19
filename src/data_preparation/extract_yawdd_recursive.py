import os, cv2, glob

RAW_YAWDD = "data/raw/yawdd"
OUT_PATH = "data/raw/yawdd_frames"
os.makedirs(OUT_PATH, exist_ok=True)

video_paths = glob.glob(os.path.join(RAW_YAWDD, "**/*.avi"), recursive=True)
print(f"Found {len(video_paths)} videos in YawDD dataset")

for vid in video_paths:
    name = os.path.splitext(os.path.basename(vid))[0]
    label = "yawn" if "yawn" in name.lower() else "no_yawn"
    out_dir = os.path.join(OUT_PATH, label, name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 10 == 0:  # Save every 10th frame
            frame_name = os.path.join(out_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_name, frame)
        frame_idx += 1
    cap.release()
    print(f"✅ Extracted frames from {vid} → {out_dir}")
