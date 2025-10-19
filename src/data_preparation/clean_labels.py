import os
import cv2

DATA_DIR = "data/merged"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

def validate_and_copy(subfolder):
    src = os.path.join(DATA_DIR, subfolder)
    dst = os.path.join(CLEAN_DIR, subfolder)
    os.makedirs(dst, exist_ok=True)
    count = 0
    for f in os.listdir(src):
        if not f.lower().endswith((".jpg",".png",".jpeg")):
            continue
        path = os.path.join(src,f)
        img = cv2.imread(path)
        if img is None or img.size == 0:
            continue
        h,w = img.shape[:2]
        if h < 30 or w < 30:  # skip tiny images
            continue
        cv2.imwrite(os.path.join(dst, f), img)
        count += 1
    print(f"Cleaned {count} images in {subfolder}")

for folder in os.listdir(DATA_DIR):
    validate_and_copy(folder)

print("All images validated and copied to", CLEAN_DIR)
