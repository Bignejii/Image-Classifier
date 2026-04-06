import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Hardware Status ---")
print(f"Active Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Load ResNet50 Intelligence
print("\n[Status] Loading ResNet50...")
model = models.resnet50(weights='DEFAULT').to(device)
model.eval()

def predict(image_path):
    # 3. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(img_t)
    
    # 5. Fetch Labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url).text.splitlines()

    # 6. Top 3 Probabilities
    percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100
    confidences, indices = torch.topk(output, 3)

    print(f"\n--- Results for: {image_path} ---")
    for i in range(3):
        idx = indices[0][i].item()
        prob = percentages[idx].item()
        print(f"Rank {i+1}: {labels[idx]} | {prob:.2f}%")

if __name__ == "__main__":
    try:
        predict("test.jpg") # Make sure 'test.jpg' exists
    except Exception as e:
        print(f"\n[Error]: {e}")