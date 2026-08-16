"""
Random Forest Classifier Pipeline
Project 5: Customer Churn Classification

This script performs:
1. Data loading and initial exploration.
2. Training and testing set preparation using the provided splits.
3. Preprocessing (Imputation & scaling for numericals, encoding for categoricals).
4. Baseline Random Forest model training and evaluation.
5. Hyperparameter tuning using GridSearchCV with 5-fold cross-validation.
6. Model evaluation, performance comparison, and feature importance analysis.
7. Outputting of performance metrics and saving of visualization plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configure styles
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data():
    """
    Load the training and testing customer churn datasets from CSV files.
    """
    print("--- 1. Loading Dataset ---")
    train_path = os.path.join(DATA_DIR, "churn-bigml-80.csv")
    test_path = os.path.join(DATA_DIR, "churn-bigml-20.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Train and Test datasets loaded successfully from local CSVs!")
        return train_df, test_df
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        raise e

def main():
    # 1. Load Data
    train_df, test_df = load_data()
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # 2. Select Features and Target
    num_cols = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes',
        'Total intl calls', 'Total intl charge', 'Customer service calls'
    ]
    cat_cols = ['State', 'Area code', 'International plan', 'Voice mail plan']
    selected_features = num_cols + cat_cols
    target = 'Churn'

    # Check if target is present
    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found in training dataset.")

    # Process splits
    X_train = train_df[selected_features].copy()
    y_train = train_df[target].astype(int)
    X_test = test_df[selected_features].copy()
    y_test = test_df[target].astype(int)

    # Convert Area code to string
    X_train['Area code'] = X_train['Area code'].astype(str)
    X_test['Area code'] = X_test['Area code'].astype(str)

    print("\n--- 2. Dataset Information ---")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print("\nMissing values in training:")
    print(X_train.isnull().sum())

    # 3. Preprocessing Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    print("\n--- 3. Preprocessing Data ---")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    num_feature_names = num_cols
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_feature_names + cat_feature_names
    print(f"Total features after preprocessing: {len(feature_names)}")

    # 4. Train Baseline Model
    print("\n--- 4. Training Baseline Model ---")
    baseline_rf = RandomForestClassifier(random_state=42)
    baseline_rf.fit(X_train_preprocessed, y_train)
    y_pred_baseline = baseline_rf.predict(X_test_preprocessed)

    baseline_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_baseline),
        'Precision': precision_score(y_test, y_pred_baseline, zero_division=0),
        'Recall': recall_score(y_test, y_pred_baseline, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_baseline, zero_division=0)
    }

    print("Baseline Model Performance:")
    for m, val in baseline_metrics.items():
        print(f" - {m}: {val:.4f}")

    # Plot & Save Baseline Confusion Matrix
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'])
    plt.title('Baseline Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'baseline_confusion_matrix.png'), dpi=300)
    plt.close()

    # 5. Hyperparameter Tuning using Cross-Validation
    print("\n--- 5. Hyperparameter Tuning (Grid Search with 5-fold CV) ---")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_preprocessed, y_train)
    print("\nBest Hyperparameters Found:")
    for param, val in grid_search.best_params_.items():
        print(f" - {param}: {val}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    # 6. Evaluate Tuned Model
    print("\n--- 6. Evaluating Tuned Model ---")
    tuned_rf = grid_search.best_estimator_
    y_pred_tuned = tuned_rf.predict(X_test_preprocessed)

    tuned_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_tuned),
        'Precision': precision_score(y_test, y_pred_tuned, zero_division=0),
        'Recall': recall_score(y_test, y_pred_tuned, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_tuned, zero_division=0)
    }

    print("Tuned Model Performance:")
    for m, val in tuned_metrics.items():
        print(f" - {m}: {val:.4f}")

    print("\nClassification Report (Tuned Model):")
    print(classification_report(y_test, y_pred_tuned))

    # Plot & Save Tuned Confusion Matrix
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'])
    plt.title('Tuned Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tuned_confusion_matrix.png'), dpi=300)
    plt.close()

    # 7. Compare Models
    print("\n--- 7. Comparing Baseline vs. Tuned ---")
    comparison_df = pd.DataFrame({
        'Metric': list(baseline_metrics.keys()),
        'Baseline': list(baseline_metrics.values()),
        'Tuned': list(tuned_metrics.values())
    })
    print(comparison_df.to_string(index=False))

    # Save comparison plot
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
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
    plt.close()

    # 8. Feature Importance Analysis
    print("\n--- 8. Feature Importance Analysis ---")
    importances = tuned_rf.feature_importances_
    feat_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feat_importance_df.head(15).to_string(index=False))

    # Save feature importance plot (top 15 features for clarity due to state one-hot encoding cardinality)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_importance_df.head(15), x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Top 15 Feature Importance Analysis (Tuned Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importances.png'), dpi=300)
    plt.close()
    print("\nAll plots have been saved to the workspace.")

if __name__ == '__main__':
    main()
