import os
from ultralytics import YOLO
from pose import process_all_dirs
from classifier import filter_crops

model = YOLO("yolo11s-pose.pt")

INTPUT_DIR = "train/images"
OUTPUT_DIR_CROP = "out/crop"
OUTPUT_DIR_CLASSIFER = "out/classifer"
CLASSIFIER_PATH = "models/model_weights.pth"

RESULT = "out/result.json"
MODEL = "models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
M2 = "models/updated_checkpoint.pth"

print("Processing Stage 1 : Pose Detection")
process_all_dirs(INTPUT_DIR, OUTPUT_DIR_CROP, model)

print("Processing Stage 2 : Classifier")
filter_crops(OUTPUT_DIR_CROP, CLASSIFIER_PATH, OUTPUT_DIR_CLASSIFER)

print("Processing Stage 3 : Moving to imgs")
command = """find out/classifer -type f -iname '*.jpg' -exec mv {} out/imgs \;"""
os.system(command)

print("Processing Stage 4 : Running STR")
command = f"python3 str.py  {MODEL}\
    --data_root=out/ --batch_size=1 --inference --result_file {RESULT}"
os.system(command)
print("Done predict numbers")