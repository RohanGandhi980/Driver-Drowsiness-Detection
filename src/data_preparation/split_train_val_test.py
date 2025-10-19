import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

ANNOT_DIR = "data/annotations"
SPLIT_DIR = "data/splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

def split_dataset(csv_path, split_dir, label_map):
    df = pd.read_csv(csv_path)
    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    splits = {"train": train, "val": val, "test": test}

    for split_name, split_df in splits.items():
        for label_name, label_val in label_map.items():
            out_dir = os.path.join(split_dir, split_name, label_name)
            os.makedirs(out_dir, exist_ok=True)

            subset = split_df[split_df["label"] == label_val]
            for _, row in subset.iterrows():
                if os.path.exists(row["path"]):
                    shutil.copy(row["path"], out_dir)

        print(f"{split_name}: {len(split_df)} images")

    print(f"✅ Split complete for {os.path.basename(csv_path).split('_')[0]} dataset")

# Split eyes
split_dataset(
    os.path.join(ANNOT_DIR, "eye_labels.csv"),
    os.path.join(SPLIT_DIR, "eyes"),
    {"open": 1, "closed": 0}
)

# Split mouth
split_dataset(
    os.path.join(ANNOT_DIR, "mouth_labels.csv"),
    os.path.join(SPLIT_DIR, "mouth"),
    {"yawn": 1, "no_yawn": 0}
)
