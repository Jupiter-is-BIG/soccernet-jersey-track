import os
import json
import shutil

from ultralytics import YOLO
from pose import process_all_dirs
from classifier import filter_crops
from prediction_sampler import process_jersey_id_predictions

model = YOLO("yolo11s-pose.pt")

INTPUT_DIR = "train/images"
OUTPUT_DIR_CROP = "out/crop"
DEST_DIR = "out/imgs"
OUTPUT_DIR_CLASSIFER = "out/classifer"
CLASSIFIER_PATH = "models/legibility_classifier_1090.pth"
NUMBER_OF_TRACKLETS = sum(os.path.isdir(os.path.join(INTPUT_DIR, d)) for d in os.listdir(INTPUT_DIR))

RESULT = "out/result.json"
SAMPLE_RESULT = "out/final.json"
MODEL = "models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
M2 = "models/updated_checkpoint.pth"

# print("Processing Stage 1 : Pose Detection")
# process_all_dirs(INTPUT_DIR, OUTPUT_DIR_CROP, model)

# print("Processing Stage 2 : Classifier")
# filter_crops(OUTPUT_DIR_CROP, CLASSIFIER_PATH, OUTPUT_DIR_CLASSIFER)

# print("Processing Stage 3 : Moving to imgs")
# os.makedirs(DEST_DIR, exist_ok=True)
# for root, dirs, files in os.walk(OUTPUT_DIR_CROP):
#     for file in files:
#         if file.lower().endswith('.jpg'):
#             file_path = os.path.join(root, file)
#             shutil.move(file_path, os.path.join(DEST_DIR, file))
# print("Files moved successfully.")

print("Processing Stage 4 : Running STR")
command = f"python3 str.py  {MODEL}\
    --data_root=out/ --batch_size=1 --inference --result_file {RESULT}"
os.system(command)
print("Done running STR model")

# print("Processing Step 5 : Sampling best inference from predictions")
# results_dict, analysis_results = process_jersey_id_predictions(RESULT, useBias=True)
# for i in range(NUMBER_OF_TRACKLETS):
#     if str(i) not in results_dict:
#         results_dict[str(i)] = -1
#     else:
#         results_dict[str(i)] = int(results_dict[str(i)])
# with open(SAMPLE_RESULT, "w") as file:
#     json.dump(
#         {str(k): results_dict[str(k)] for k in sorted(map(int, results_dict.keys()))},
#         file, 
#         indent=4
#     )
# print("Done predict numbers")