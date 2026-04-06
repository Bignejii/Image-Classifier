# AI Image Classifier (PyTorch + CUDA)

A high-performance image classification project built with **Python 3.12** and **PyTorch**. This project leverages the **NVIDIA RTX 4060 Ti** GPU to perform fast deep learning inference using the **ResNet50** architecture.

## 🚀 Hardware & Performance
- **GPU:** NVIDIA GeForce RTX 4060 Ti (8GB VRAM)
- **CPU:** AMD Ryzen 5 5600G
- **RAM:** 16GB @ 3800MHz
- **Performance:** Capable of processing 100M+ operations in ~1.1 seconds.

---

## 📁 Project Structure
```text
Image-Classifier/
├── venv/                   # Python Virtual Environment
├── images/                 # Folder for batch processing multiple images
├── start_ai.py             # Script for single image prediction (Top 3 results)
├── batch_analyzer.py       # Script to analyze all images in 'images/' folder
├── requirements.txt        # Project dependencies
└── test.jpg                # Sample image for testing



Setup & Installation
Clone the project or create the folder structure.

Install Dependencies (Ensure you use the CUDA-enabled torch):

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install pillow requests



Verify GPU connection:
Run start_ai.py. It should display Active Device: NVIDIA GeForce RTX 4060 Ti.