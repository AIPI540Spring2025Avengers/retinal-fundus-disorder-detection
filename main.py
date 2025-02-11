import streamlit as st
import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from skimage.feature import local_binary_pattern
from scripts.naive.model import classify_fundus_image

# Class names for different models
NAIVE_CLASS_NAMES = [
    'Cataract', 'Dry AMD', 'Glaucoma', 'Mild DR', 'Moderate DR',
    'Normal Fundus', 'Pathological Myopia', 'Proliferate DR',
    'Severe DR', 'Uncertain - Further Review Needed', 'Wet AMD'
]

DL_CLASS_NAMES = [
    '1.Dry AMD', '2.Wet AMD', '3.Mild DR', '4.Moderate DR', '5.Severe DR',
    '6.Proliferate DR', '7.Cataract', '8.Hypersensitive Retinopathy',
    '9.Pathological Myopia', '10.Glaucoma', '11.Normal Fundus'
]

class TraditionalModelHandler:
    """Handler for traditional ML model predictions with label encoder"""
    
    def __init__(self, model_path: str, encoder_path: str):
        """
        Initialize traditional model handler.
        
        Args:
            model_path: Path to saved pipeline pickle file
            encoder_path: Path to saved label encoder
        """
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed image"""
        # Convert color space and preprocess
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Color features
        color_hists = [cv2.calcHist([lab], [c], None, [64], [0, 256]).flatten() 
                      for c in range(3)]
        color_hists = np.concatenate([h / (h.sum() + 1e-6) for h in color_hists])
        
        # Texture features using scikit-image
        lbp = local_binary_pattern(gray, 16, 3, method='uniform')  # Original implementation
        lbp_hist, _ = np.histogram(lbp, bins=18, range=(0, 18))
        lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-6)
        
        # Edge features
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        vessel_mag = np.sum(np.sqrt(sobel_x**2 + sobel_y**2))
        
        return np.concatenate((color_hists, lbp_hist, [vessel_mag]))

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Make prediction using traditional ML model.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            np.ndarray: Probability distribution over classes
        """
        features = self._extract_features(image)
        return self.pipeline.predict_proba([features])[0]

class DeepLearningModelHandler:
    """Handler for deep learning model predictions with directory-based labels"""
    
    def __init__(self, model_path: str):
        """
        Initialize deep learning model handler.
        
        Args:
            model_path: Path to saved PyTorch model weights
        """
        self.model = models.efficientnet_b4()
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(1792),
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 11)
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Make prediction using deep learning model.
        
        Args:
            image: PIL Image object
            
        Returns:
            np.ndarray: Probability distribution over classes
        """
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
        return torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
    
@st.cache_resource
def load_traditional_model_handler():
    """Loads and caches the traditional model handler."""
    return TraditionalModelHandler(
        'models/TRADML_model.pkl',
        'models/TRADML_encoder.pkl'
    )

@st.cache_resource
def load_deep_learning_model_handler():
    """Loads and caches the deep learning model handler."""
    return DeepLearningModelHandler('models/efficientnet_b4_retinal.pth')

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Fundus Image Analysis", layout="wide")
    st.title("Retinal Fundus Image Classification")
    
    # Initialize models
    trad_handler = load_traditional_model_handler()
    dl_handler = load_deep_learning_model_handler()
    
    # File upload
    uploaded_file = st.file_uploader("Upload fundus image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert to OpenCV format for traditional model
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with col2:
            # Naive model prediction using imported function
            naive_pred = classify_fundus_image(image_cv)
            st.subheader(f"Rule-Based Diagnosis: {naive_pred}")

            # Traditional model predictions
            trad_probs = trad_handler.predict(image_cv)
            st.subheader("Traditional Model Probabilities")
            display_probability_grid(trad_probs, trad_handler.encoder.classes_)
            
            # Deep learning predictions
            dl_probs = dl_handler.predict(image)
            st.subheader("Deep Learning Model Probabilities")
            display_probability_grid(dl_probs, DL_CLASS_NAMES)

def display_probability_grid(probs: np.ndarray, class_names: list):
    """Display probabilities in numerical order based on label prefixes"""
    # Extract numerical prefixes and sort
    sorted_pairs = sorted(
        zip(class_names, probs),
        key=lambda x: int(x[0].split('.')[0])  # Split on first . and convert to int
    )
    
    # Display in grid
    for i in range(0, len(sorted_pairs), 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j
            if idx < len(sorted_pairs):
                class_name, prob = sorted_pairs[idx]
                with cols[j]:
                    st.metric(
                        label=class_name,
                        value=f"{prob:.2%}"
                    )

if __name__ == "__main__":
    main()

## Above code generated using the DeepSeek R1 model in Perplexity, and then tweaked. 
## The prompt provided the model training code for the three models and requested a Streamlit application which would take an image upload
## from the user and then run the three models. Subsequent prompting iterations corrected label decoding inconsistencies between the models.
