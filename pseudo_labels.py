import os
import shutil
import easyocr
import pandas as pd
from tqdm import tqdm

INPUT_PATH = "soccernet-train-cropped"
OUTPUT_PATH = "class_train"
OUTPUT_CSV = "labels_train.csv"

os.makedirs(OUTPUT_PATH, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=True)

def detect_number(image_path):
    try:
        results = reader.readtext(image_path, detail=0)
        for text in results:
            if any(char.isdigit() for char in text):
                return 1 
        return 0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return -1

image_labels = []

subdirs = [os.path.join(INPUT_PATH, d) for d in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH, d))]
total_files = sum(len(files) for _, _, files in os.walk(INPUT_PATH))

global_progress = tqdm(total=total_files, desc="Overall Progress", position=0, leave=True)

for subdir in subdirs:
    subdir_name = os.path.basename(subdir)
    files = [f for f in os.listdir(subdir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    with tqdm(total=len(files), desc=f"Processing {subdir_name}", position=1, leave=False) as pbar:
        for image_name in files:
            image_path = os.path.join(subdir, image_name)
            
            # Running OCR Model for detction here :)
            label = detect_number(image_path)
            relative_name = f"{subdir_name}_{image_name}"
            image_labels.append({"image": relative_name, "label": label})
            
            if label == 1:
                output_subdir = os.path.join(OUTPUT_PATH, subdir_name)
                os.makedirs(output_subdir, exist_ok=True)
                shutil.copy(image_path, os.path.join(output_subdir, image_name))
            
            pbar.update(1)
            global_progress.update(1)

global_progress.close()

df = pd.DataFrame(image_labels)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Labeling complete! Saved labels to {OUTPUT_CSV}")
print(f"Detected images saved in {OUTPUT_PATH}")