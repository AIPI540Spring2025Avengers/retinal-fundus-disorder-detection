import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Script location
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")  # Processed features
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models")  # Directory to save model
RANDOM_STATE = 42

# LOAD DATA
def load_features(split):
    """Loads features and labels from preprocessed dataset."""
    feature_path = os.path.join(PROCESSED_DIR, f"features_{split}.pkl")
    label_path = os.path.join(PROCESSED_DIR, f"labels_{split}.pkl")
    
    # Check if files exist
    if not os.path.exists(feature_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing processed files for {split}. Run preprocessing first.")

    # Load features and labels
    with open(feature_path, "rb") as f:
        X = pickle.load(f)

    with open(label_path, "rb") as f:
        y = pickle.load(f)

    return np.array(X, dtype=np.float32), np.array(y)

# BUILD ML PIPELINE
def build_pipeline():
    """Pipeline with scaling, PCA, and SVM."""
    # Pipeline steps
    return Pipeline([
        ('scaler', StandardScaler()),  
        ('pca', PCA(n_components=0.98, random_state=RANDOM_STATE)),  
        ('svm', SVC(kernel='rbf', C=20.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
    ])

# TRAINING AND EVALUATION
def main():
    """Loads data, trains the model, evaluates, and saves the trained model."""
    # Load training & validation data
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)  

    # Train the model
    print("Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap="Blues", xticks_rotation=90)
    plt.show()

    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "TRADML_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")


# MAIN FUNCTION
if __name__ == "__main__":
    main()
