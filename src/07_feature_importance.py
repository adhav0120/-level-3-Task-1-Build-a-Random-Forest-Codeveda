import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def main():
    print("=========================================")
    print("STEP 7: Feature Importance Analysis")
    print("=========================================")
    
    # Load tuned model
    try:
        tuned_rf = joblib.load(os.path.join(MODELS_DIR, "tuned_model.joblib"))
    except FileNotFoundError:
        print("Error: tuned_model.joblib not found. Run 06_model_evaluation.py first!")
        return

    # Load feature names
    try:
        with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
            feature_names = json.load(f)
    except FileNotFoundError:
        print("Error: feature_names.json not found. Run 03_preprocessing.py first!")
        return

    # Extract importances
    importances = tuned_rf.feature_importances_
    
    # Construct DataFrame
    feat_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance Rankings:")
    print(feat_importance_df.to_string(index=False))

    # Plot importances
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_importance_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Importance Analysis (Tuned Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    importance_plot_path = os.path.join(PLOTS_DIR, "feature_importances.png")
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()
    
    print(f"\nFeature Importance chart saved as: {importance_plot_path}")

if __name__ == '__main__':
    main()
