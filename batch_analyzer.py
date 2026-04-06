import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import requests

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='DEFAULT').to(device)
model.eval()

# Load Labels
labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def analyze_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created. Put images inside and run again.")
        return

    print(f"--- Batch Processing on {device} ---")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_t)
                    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
                    _, index = torch.max(output, 1)

                print(f"File: {filename:15} | Result: {labels[index[0]]:15} | {prob[index[0]]:.2f}%")
            except Exception as e:
                print(f"Error {filename}: {e}")

if __name__ == "__main__":
    analyze_folder("images")