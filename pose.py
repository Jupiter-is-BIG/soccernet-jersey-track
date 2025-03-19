import os

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def crop(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    images = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in tqdm(images, leave=False):

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        img_width = image.shape[1]
        img_height = image.shape[0]

        results = model(image_path, verbose=False)

        if (
            len(results[0].keypoints) != 1
        ):  # we only continue if we detect exactly one person
            continue

        keypoints = results[0].keypoints.xy.cpu().numpy()[0]

        if keypoints is None or len(keypoints) < 13:
            continue

        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]

        # Bounding Box
        x_min = (
            int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])) - 2
        )
        x_max = (
            int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])) + 2
        )
        y_min = int(min(left_shoulder[1], right_shoulder[1]))
        y_max = int(max(left_hip[1], right_hip[1]))

        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, img_width), min(y_max, image.shape[0])

        if x_max - x_min < 0.25 * img_width or y_max - y_min < 0.1 * img_height:
            continue

        cropped_tshirt = image[y_min:y_max, x_min:x_max]

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped_tshirt)


def process_all_dirs(input_root, output_root, model):
    os.makedirs(output_root, exist_ok=True)

    subdirs = [
        d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))
    ]

    for subdir in tqdm(subdirs, desc="Processing directories"):
        input_dir = os.path.join(input_root, subdir)
        output_dir = os.path.join(output_root, subdir)

        crop(input_dir, output_dir, model)

    print(f"Processing complete. Cropped images saved in: {output_root}")