import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    print("=========================================")
    print("STEP 4: Baseline Model Training & Evaluation")
    print("=========================================")
    
    # Load preprocessed arrays
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train_preprocessed.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test_preprocessed.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("Error: Preprocessed data files not found. Please run 03_preprocessing.py first!")
        return

    # Train baseline model
    print("Training baseline Random Forest with default parameters...")
    baseline_rf = RandomForestClassifier(random_state=42)
    baseline_rf.fit(X_train, y_train)

    # Predict
    y_pred = baseline_rf.predict(X_test)

    # Evaluate
    baseline_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0)
    }

    print("\nBaseline Model Performance on Test Set:")
    for metric, score in baseline_metrics.items():
        print(f" - {metric}: {score:.4f}")

    print("\nClassification Report (Baseline):")
    print(classification_report(y_test, y_pred))

    # Save baseline metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f)

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Baseline Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm_path = os.path.join(PLOTS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"\nBaseline Confusion Matrix saved as: {cm_path}")
    print(f"Baseline metrics stored in: {os.path.join(MODELS_DIR, 'baseline_metrics.json')}")

if __name__ == '__main__':
    main()
