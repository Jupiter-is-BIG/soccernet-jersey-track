import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
def filter_crops(dir_path, weight_path, out_dir):
    model = BinaryClassifier()
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # To convert tensor back to a PIL image (denormalizing)
    to_pil = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Reverse normalization
        transforms.ToPILImage()
    ])

    os.makedirs(out_dir, exist_ok=True)
    
    for subdir_name in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir_name)
        if os.path.isdir(subdir_path):
            out_subdir = os.path.join(out_dir, subdir_name)
            os.makedirs(out_subdir, exist_ok=True)
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    with Image.open(file_path) as img:
                        if img.width < 15 or img.height < 15:
                            continue
                        img_transformed = transform(img).unsqueeze(0)  # Add batch dimension
                        with torch.no_grad():
                            output = model(img_transformed)
                            prediction = output.item()
                        
                        if prediction > 0.5:
                            # Convert back to PIL image before saving
                            img_resized = to_pil(img_transformed.squeeze(0))  # Remove batch dim
                            img_resized.save(os.path.join(out_subdir, filename))
