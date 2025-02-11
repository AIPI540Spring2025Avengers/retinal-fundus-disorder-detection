import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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


# TRAINING AND EVALUATION FUNCTION
def train_and_evaluate(X_train, y_train, X_test, y_test, round_name, le=None):
    """Trains and evaluates the model."""
    # Train model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation ({round_name}):")
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report if label encoder is provided
    if le:
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    return pipeline, y_pred

# MAIN FUNCTION
def main():
    """Loads data, trains the model, evaluates, and saves the trained model."""
    
    # Load datasets
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    X_test, y_test = load_features("test")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    # Train on train set → Evaluate on val
    print("\nTraining on train set...")

    train_and_evaluate(X_train, y_train, X_val, y_val, "Train → Val", le)

    # Retrain on train + val → Evaluate on test
    print("\nRetraining on train + val set...")
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    final_pipeline, y_pred_test = train_and_evaluate(X_train_val, y_train_val, X_test, y_test, "Train+Val → Test", le)

    # Print confusion matrix
    print("\nFinal Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap="Blues", xticks_rotation=90)
    plt.show()

    # Save final model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "TRADML_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "TRADML_encoder.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"Final model and encoder saved at: {MODEL_DIR}")

# RUN MAIN FUNCTION
if __name__ == "__main__":
    main()

## ChatGPT-4o was used to produce above code on 2/6/25.
## Goal was to reorganize previous code notebook into an organized and modular script format
## Prompt was "turn this notebook into a cohesive and organized script"
