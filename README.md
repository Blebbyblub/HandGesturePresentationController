# Hand Gesture Presentation Controller

A real-time hand gesture recognition system for controlling presentation slides using Computer Vision and Machine Learning techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-89.02%25-brightgreen.svg)

## � Download

**No installation required!** Download the standalone executable:

👉 **[Download HandGestureController.exe](https://github.com/yourusername/HandGesturePresentationController/releases/latest/download/HandGestureController.exe)** 👈

> Just download, run, and control your presentations with hand gestures!

## �📋 Project Overview

This project implements a touchless presentation control system using hand gestures captured through a webcam. The system recognizes two gestures:
- **👉 Next** - Navigate to next slide (swipe right gesture)
- **👈 Previous** - Navigate to previous slide (swipe left gesture)

Developed for **COMP6826001 - Computer Vision** Final Project at BINUS University.

## ✨ Key Features

- **Real-time Processing**: 30+ FPS on standard hardware
- **Low Latency**: ~14.11ms average inference time
- **High Accuracy**: 89.02% classification accuracy
- **Robust Detection**: Works in various lighting conditions
- **Cooldown System**: Prevents accidental repeated triggers

## 🔬 Technical Pipeline

```
Webcam Input → MediaPipe Hands → ROI Extraction → Canny Edge Detection 
→ HOG Feature Extraction → SVM Classification → Keyboard Simulation
```

### Components:
1. **Hand Detection**: MediaPipe Hands for robust hand landmark detection
2. **Edge Detection**: Canny algorithm (thresholds: 50/150) for edge extraction
3. **Feature Extraction**: HOG (9 orientations, 16×16 cells, 2×2 blocks)
4. **Classification**: SVM with RBF kernel (C=10, gamma=scale)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 89.02% |
| Precision (Macro) | 89.18% |
| Recall (Macro) | 89.02% |
| F1-Score (Macro) | 89.01% |
| Inference Time | 14.11ms |
| Frame Rate | 30+ FPS |

## 📁 Repository Structure

```
HandGesturePresentationController/
│
├── controller.py           # Main application (run this!)
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── artifacts/             # Trained models and features
│   ├── gesture_svm_v3.pkl # Final SVM model
│   ├── canny_hog_*.npz    # HOG features (Canny version)
│   └── hog_*.npz          # HOG features (original)
│
├── notebooks/             # Jupyter notebooks for training
│   ├── new_imageClassify_V3Tuned.ipynb  # Final tuned model
│   ├── new_imageClassify_V3(Raenault).ipynb
│   ├── new_imageClassify_V2canny.ipynb
│   ├── new_imageClassify.ipynb
│   └── ImageClassify.ipynb
│
├── scripts/               # Utility scripts
│   ├── build_exe.py       # PyInstaller build script
│   └── debug_controller.py # Debug version with verbose output
│
├── dataset_final/         # Processed dataset (train/valid/test)
├── Dataset/               # Raw images (Back/Next)
└── Dataset_roboflow/      # Roboflow augmented dataset
```

## 🚀 Quick Start

### Option 1: Download Executable (Easiest)

1. Download `HandGestureController.exe` from [Releases](https://github.com/yourusername/HandGesturePresentationController/releases)
2. Double-click to run
3. Allow webcam access when prompted
4. Start controlling your presentations!

### Option 2: Run from Source

#### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows OS (for keyboard simulation)

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Blebbyblub/HandGesturePresentationController.git
   cd HandGesturePresentationController
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the controller**
   ```bash
   python controller.py
   ```

2. **Controls**
   - `Q` - Quit the application
   - `D` - Toggle debug mode
   - `C` - Toggle cooldown display

3. **Gestures**
   - Show your hand with fingers pointing **right** → Next slide
   - Show your hand with fingers pointing **left** → Previous slide

## 📦 Dependencies

```
numpy==1.26.4
opencv-python==4.8.1.78
mediapipe==0.10.14
scikit-learn==1.4.2
scikit-image==0.22.0
pyautogui==0.9.54
joblib==1.3.2
```

## 📈 Dataset

| Split | Images | Description |
|-------|--------|-------------|
| Training | 375 | Augmented via Roboflow |
| Validation | 16 | For hyperparameter tuning |
| Test | 16 | Final evaluation |
| **Total** | **407** | From 82 original images |

### Augmentation Techniques:
- Horizontal flip
- Rotation (±15°)
- Brightness adjustment (±25%)
- Blur (up to 2.5px)

## 🧪 Model Training

Training notebooks are located in the `notebooks/` folder. The final model was trained using:

```python
# Best hyperparameters from GridSearchCV
svm = SVC(kernel='rbf', C=10, gamma='scale')
```

To retrain:
1. Open `notebooks/new_imageClassify_V3Tuned.ipynb`
2. Run all cells
3. Model saves to `artifacts/gesture_svm_v3.pkl`



## 📄 License

This project is developed for educational purposes as part of BINUS University coursework.

---
