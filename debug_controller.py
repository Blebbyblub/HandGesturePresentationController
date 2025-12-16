"""
DEBUG SCRIPT FOR HAND GESTURE SVM (CANNY + HOG)

What this script does:
1. Loads the trained model bundle (.pkl)
2. Prints model + config info
3. Loads sample images from dataset
4. Runs Canny + HOG feature extraction
5. Runs prediction + probabilities
6. Prints detailed debug output

This does NOT use controller.py
"""

from pathlib import Path
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path.cwd()

MODEL_PATH = BASE_DIR / "artifacts" / "gesture_svm_v2_Wcanny.pkl"

# Use cropped dataset (preferred)
DATASET_ROOT = BASE_DIR / "dataset_final"
FALLBACK_DATASET = BASE_DIR / "Dataset"

TARGET_IMAGE_SIZE = (128, 128)

CANNY_PARAMS = {
    "threshold1": 50,
    "threshold2": 150,
    "apertureSize": 3,
    "L2gradient": True,
}

HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "transform_sqrt": True,
    "block_norm": "L2-Hys",
    "feature_vector": True,
}

ALLOWED_EXTS = {".jpg", ".png", ".jpeg"}

# =========================
# UTILITIES
# =========================

def resolve_dataset_root():
    if DATASET_ROOT.exists() and any(DATASET_ROOT.rglob("*.jpg")):
        return DATASET_ROOT
    if FALLBACK_DATASET.exists() and any(FALLBACK_DATASET.rglob("*.jpg")):
        print("⚠️ Using fallback dataset")
        return FALLBACK_DATASET
    raise FileNotFoundError("No dataset found")

def load_image(path):
    img = Image.open(path).convert("L").resize(TARGET_IMAGE_SIZE)
    return np.asarray(img, dtype=np.uint8)

def extract_canny_hog(image_gray):
    edges = cv2.Canny(
        image_gray,
        CANNY_PARAMS["threshold1"],
        CANNY_PARAMS["threshold2"],
        apertureSize=CANNY_PARAMS["apertureSize"],
        L2gradient=CANNY_PARAMS["L2gradient"],
    )

    edges_norm = edges.astype(np.float32) / 255.0

    hog_feat = hog(edges_norm, **HOG_PARAMS)
    canny_flat = edges.flatten().astype(np.float32) / 255.0

    return np.concatenate([hog_feat, canny_flat])

# =========================
# MAIN DEBUG ROUTINE
# =========================

def main():
    print("=" * 60)
    print("HAND GESTURE MODEL DEBUGGER")
    print("=" * 60)

    # -------- LOAD MODEL --------
    print("\n[1] Loading model...")
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    target_size = bundle.get("target_size", TARGET_IMAGE_SIZE)
    label_map = bundle.get("label_map", {})

    print("✓ Model loaded")
    print(f"  Kernel      : {model.kernel}")
    print(f"  Classes     : {model.classes_}")
    print(f"  Target size : {target_size}")
    print(f"  Label map   : {label_map}")

    # -------- LOAD DATA --------
    dataset_root = resolve_dataset_root()
    print(f"\n[2] Using dataset: {dataset_root}")

    image_paths = list(dataset_root.rglob("*.jpg"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ALLOWED_EXTS]

    print(f"✓ Found {len(image_paths)} images")

    # -------- RUN INFERENCE --------
    print("\n[3] Running inference...\n")

    correct = 0
    total = 0

    for img_path in image_paths[:30]:  # limit for debug
        true_label = img_path.parent.name.lower()

        img_gray = load_image(img_path)
        feature = extract_canny_hog(img_gray)

        feature = feature.reshape(1, -1)

        probs = model.predict_proba(feature)[0]
        pred_idx = np.argmax(probs)
        pred_label = model.classes_[pred_idx]
        confidence = probs[pred_idx]

        is_correct = pred_label == true_label
        correct += int(is_correct)
        total += 1

        print(f"Image: {img_path.name}")
        print(f"  True label : {true_label}")
        print(f"  Prediction : {pred_label}")
        print(f"  Confidence : {confidence:.3f}")
        print(f"  Probs      : {dict(zip(model.classes_, probs.round(3)))}")
        print(f"  Result     : {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print("-" * 50)

        # Optional visualization
        plt.figure(figsize=(3, 3))
        plt.imshow(img_gray, cmap="gray")
        plt.title(f"{pred_label} ({confidence:.2f})")
        plt.axis("off")
        plt.show()

    # -------- SUMMARY --------
    print("\n[4] SUMMARY")
    print("=" * 30)
    print(f"Accuracy (debug subset): {correct}/{total} = {correct/total:.2%}")

    print("\nDEBUG COMPLETE ✅")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()
