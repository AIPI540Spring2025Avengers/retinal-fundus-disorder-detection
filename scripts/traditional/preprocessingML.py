import os
import random
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern

# CONFIGURATION

DATA_PATH = os.path.join("data", "raw", "Retinal Fundus Images")  # Raw dataset location
OUTPUT_DIR = os.path.join("data", "processed")  # Processed dataset location
IMG_SIZE = (512, 512)  # Resize images 
SAMPLES_PER_CLASS = 250  # class samples

FEATURE_PARAMS = {
    'color_bins': 64,
    'lbp': {
        'radius': 3,
        'n_points': 16,
        'method': 'uniform'
    }
}

# DATA LOADING & PREPROCESSING
def load_dataset(data_path, samples_per_class=None):
    """Loads images and their labels from folders."""
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    image_paths, labels = [], []

    # Load images and labels
    for cls in class_names:
        cls_path = os.path.join(data_path, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpeg'))]
        random.shuffle(files)
        if samples_per_class:
            files = files[:samples_per_class]
        for f in files:
            image_paths.append(os.path.join(cls_path, f))
            labels.append(cls)
    
    return image_paths, labels, class_names

def preprocess_image(img_path):
    """Preprocessing: grayscale, CLAHE, resize."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Resize directly to IMG_SIZE
    gray_resized = cv2.resize(gray, IMG_SIZE)
    color_resized = cv2.resize(img, IMG_SIZE)

    return gray_resized, color_resized

def extract_features(gray_img, color_img, params):
    """Extracts color, texture, and edge features."""
    # Color features
    lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
    color_hists = [cv2.calcHist([lab], [c], None, [params['color_bins']], [0, 256]).flatten() for c in range(3)]
    color_hists = np.concatenate([h / (h.sum() + 1e-6) for h in color_hists])
    # Texture features
    lbp = local_binary_pattern(gray_img, params['lbp']['n_points'], params['lbp']['radius'], params['lbp']['method'])
    lbp_hist, _ = np.histogram(lbp, bins=params['lbp']['n_points'] + 2, range=(0, params['lbp']['n_points'] + 2))
    lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-6)
    # Edge features
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    vessel_mag = np.sum(np.sqrt(sobel_x**2 + sobel_y**2))

    return np.concatenate((color_hists, lbp_hist, [vessel_mag]))

def save_features(features, labels, split):
    """Saves processed features and labels to disk."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Save features and labels
    with open(os.path.join(OUTPUT_DIR, f"features_{split}.pkl"), "wb") as f:
        pickle.dump(features, f)
    with open(os.path.join(OUTPUT_DIR, f"labels_{split}.pkl"), "wb") as f:
        pickle.dump(labels, f)

def process_and_save(split):
    """Loads, processes, and saves features for train, val, test."""
    # Load dataset
    split_path = os.path.join(DATA_PATH, split)
    image_paths, labels, _ = load_dataset(split_path, SAMPLES_PER_CLASS)
    # Process images
    features, valid_labels = [], []
    for path, label in zip(image_paths, labels):
        gray_img, color_img = preprocess_image(path)
        # Extract features
        if gray_img is not None and color_img is not None:
            features.append(extract_features(gray_img, color_img, FEATURE_PARAMS))
            valid_labels.append(label)

    save_features(features, valid_labels, split)
    print(f"{split} data processed & saved ({len(features)} samples)")

def main():
    for split in ["train", "val", "test"]:
        process_and_save(split)

if __name__ == "__main__":
    main()
