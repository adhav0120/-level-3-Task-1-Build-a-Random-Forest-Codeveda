import os
import json
import joblib
import numpy as np
import pandas as pd
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
    print("STEP 6: Tuned Model Evaluation & Comparison")
    print("=========================================")
    
    # Load preprocessed arrays
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train_preprocessed.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test_preprocessed.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("Error: Preprocessed files not found. Run 03_preprocessing.py first!")
        return

    # Load best hyperparameters
    try:
        with open(os.path.join(MODELS_DIR, "best_hyperparameters.json"), "r") as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: best_hyperparameters.json not found. Run 05_hyperparameter_tuning.py first!")
        return

    print("Instantiating tuned Random Forest Classifier with parameters:")
    print(best_params)

    # Train model
    tuned_rf = RandomForestClassifier(**best_params, random_state=42)
    tuned_rf.fit(X_train, y_train)

    # Predict
    y_pred = tuned_rf.predict(X_test)

    # Evaluate
    tuned_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0)
    }

    print("\nTuned Model Performance on Test Set:")
    for metric, score in tuned_metrics.items():
        print(f" - {metric}: {score:.4f}")

    print("\nClassification Report (Tuned Model):")
    print(classification_report(y_test, y_pred))

    # Save tuned model and metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tuned_rf, os.path.join(MODELS_DIR, "tuned_model.joblib"))
    with open(os.path.join(MODELS_DIR, "tuned_metrics.json"), "w") as f:
        json.dump(tuned_metrics, f)

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'])
    plt.title('Tuned Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'tuned_confusion_matrix.png'), dpi=300)
    plt.close()

    # Model Comparison Visuals
    try:
        with open(os.path.join(MODELS_DIR, "baseline_metrics.json"), "r") as f:
            baseline_metrics = json.load(f)
            
        comparison_df = pd.DataFrame({
            'Metric': list(baseline_metrics.keys()),
            'Baseline': list(baseline_metrics.values()),
            'Tuned': list(tuned_metrics.values())
        })
        
        print("\nModel Comparison Table:")
        print(comparison_df.to_string(index=False))

        # Comparative Plot
        comparison_melted = pd.melt(comparison_df, id_vars='Metric', var_name='Model', value_name='Score')
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=comparison_melted, x='Metric', y='Score', hue='Model', palette='Set2')
        plt.title('Performance Comparison: Baseline vs Tuned Random Forest')
        plt.ylim(0, 1.05)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.4f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points',
                            fontsize=10,
                            fontweight='semibold')
        plt.ylabel('Score')
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
        plt.close()
        
        print(f"\nComparative bar chart saved as: {os.path.join(PLOTS_DIR, 'model_comparison.png')}")
    except FileNotFoundError:
        print(f"Warning: baseline_metrics.json not found. Comparison skipped.")

if __name__ == '__main__':
    main()
