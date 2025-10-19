import os
import shutil

# Base folders
MRL_PATH   = "data/raw/mrleyedataset"
YAWDD_PATH = "data/raw/yawdd_frames"
NTHU_PATH  = "data/raw/nthuddd"
OUT_PATH   = "data/merged"

os.makedirs(OUT_PATH, exist_ok=True)
for sub in ["eyes_open", "eyes_closed", "yawn", "no_yawn"]:
    os.makedirs(os.path.join(OUT_PATH, sub), exist_ok=True)

from tqdm import tqdm

def copy_images(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    all_files = []
    # recursive search for images
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} image files in {src_dir}")
    for src in tqdm(all_files, desc=f"Copying from {os.path.basename(src_dir)}"):
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy(src, dst)

# Merge MRL Eye dataset
copy_images(os.path.join(MRL_PATH, "Open-Eyes"), os.path.join(OUT_PATH, "eyes_open"))
copy_images(os.path.join(MRL_PATH, "Close-Eyes"), os.path.join(OUT_PATH, "eyes_closed"))

# Merge YawDD mouth states
copy_images(os.path.join(YAWDD_PATH, "yawn"), os.path.join(OUT_PATH, "yawn"))
copy_images(os.path.join(YAWDD_PATH, "no_yawn"), os.path.join(OUT_PATH, "no_yawn"))

# Merge NTHU drowsy
copy_images(os.path.join(NTHU_PATH, "drowsy"), os.path.join(OUT_PATH, "eyes_closed"))
copy_images(os.path.join(NTHU_PATH, "notdrowsy"), os.path.join(OUT_PATH, "eyes_open"))

print("✅ Datasets merged successfully into:", OUT_PATH)
