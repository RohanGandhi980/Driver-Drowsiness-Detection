import os
import pandas as pd

PROC_DIR = "data/processed"
ANNOT_DIR = "data/annotations"
os.makedirs(ANNOT_DIR, exist_ok=True)

def create_labels(subdir, out_csv):
    paths, labels = [], []
    full_dir = os.path.join(PROC_DIR, subdir)
    for f in os.listdir(full_dir):
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        # infer label from filename prefix
        if f.startswith("open_"):
            labels.append(1)
        elif f.startswith("closed_"):
            labels.append(0)
        elif f.startswith("yawn_"):
            labels.append(1)
        elif f.startswith("no_yawn_"):
            labels.append(0)
        else:
            continue
        paths.append(os.path.join(full_dir, f))
    
    df = pd.DataFrame({"path": paths, "label": labels})
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(df)} entries to {out_csv}")

create_labels("eyes", os.path.join(ANNOT_DIR, "eye_labels.csv"))
create_labels("mouth", os.path.join(ANNOT_DIR, "mouth_labels.csv"))
