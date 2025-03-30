import os
import cv2
import json
import torch
from PIL import Image
from parseq.strhub.data.module import SceneTextDataModule
from ultralytics import YOLO
from torchvision import transforms
from classifier import BinaryClassifier

# Load YOLO pose model
pose_model = YOLO("yolo11s-pose.pt")
model = BinaryClassifier()
model.load_state_dict(torch.load("models/legibility_classifier_1090.pth", map_location=torch.device('cpu')))
model.eval()
parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

MODEL = "models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
RESULT = "captured_crops/result.json"

DETECTED_NUM = -1
CONFIDENCE = -1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

to_pil = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

output_dir = "captured_crops"
output_dir2 = "captured_crops/imgs"
os.makedirs(output_dir, exist_ok=True)
frame_count = 0

legible = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    results = pose_model(frame)

    crop_detected = False
    for result in results:
        if len(result.keypoints) == 1:
            keypoints = result.keypoints.xy.cpu().numpy()[0]
            if keypoints is not None and len(keypoints) >= 13:
                left_shoulder, right_shoulder = keypoints[5], keypoints[6]
                left_hip, right_hip = keypoints[11], keypoints[12]

                x_min = int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])) - 2
                x_max = int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])) + 2
                y_min = int(min(left_shoulder[1], right_shoulder[1]))
                y_max = int(max(left_hip[1], right_hip[1]))

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, frame.shape[1]), min(y_max, frame.shape[0])

                if x_max - x_min > 0.05 * frame.shape[1] and x_max - x_min < 0.65 * frame.shape[1] and y_max - y_min > 0.1 * frame.shape[0]:
                    cropped_tshirt = frame[y_min:y_max, x_min:x_max]
                    crop_path = os.path.join(output_dir, f"crop.jpg")
                    cv2.imwrite(crop_path, cropped_tshirt)
                    frame_count += 1
                    crop_detected = True

                cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(left_hip[0]), int(left_hip[1])), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_hip[0]), int(right_hip[1])), 5, (0, 255, 0), -1)

                if crop_detected:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
    INFO = f"Detected: {DETECTED_NUM} Confidence: {CONFIDENCE:.2f}"
    COLOR = (255, 0, 0) if legible else (0, 0, 255)
    cv2.putText(frame, INFO, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2, cv2.LINE_AA)

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not crop_detected:
        continue

    legible = False
    # CALL CLASSIFIER MODEL TO CHECK IF CROP_PATH IMAGE IS GOOD OR BAD
    with Image.open(crop_path) as img:
        if img.width < 15 or img.height < 15:
            continue
        img_transformed = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_transformed)
            prediction = output.item()
        
        if prediction > 0.5:
            img_resized = to_pil(img_transformed.squeeze(0)) 
            legible = True
    
    if not legible:
        DETECTED_NUM = -1
        CONFIDENCE = -1
        continue

    img = img_transform(img_resized).unsqueeze(0)
    logits = parseq(img)
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    if confidence[0][0].item() >= 0.9:
        DETECTED_NUM = label[0]
        CONFIDENCE = confidence[0][0].item()
    else:
        DETECTED_NUM = -1
        CONFIDENCE = -1

cap.release()
cv2.destroyAllWindows()