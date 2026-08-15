"""
Random Forest Classifier Pipeline
Project 5: Random Forest Classifier

This script performs:
1. Data loading and initial exploration.
2. Stratified train-test split (leakage prevention).
3. Preprocessing (Imputation & scaling for numericals, encoding for categoricals).
4. Baseline Random Forest model training and evaluation.
5. Hyperparameter tuning using GridSearchCV with 5-fold cross-validation.
6. Model evaluation, performance comparison, and feature importance analysis.
7. Outputting of performance metrics and saving of visualization plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
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
    Load the Titanic dataset from OpenML, with fallbacks to GitHub CSV 
    and synthetic generation if internet connection is down.
    """
    print("--- 1. Loading Dataset ---")
    try:
        print("Attempting to load Titanic dataset from OpenML...")
        titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
        df = titanic.frame
        print("Dataset loaded successfully from OpenML!")
        return df
    except Exception as e:
        print(f"OpenML load failed: {e}")
        print("Falling back to downloading from public GitHub CSV...")
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            df.columns = df.columns.str.lower()
            print("Dataset loaded successfully from GitHub!")
            return df
        except Exception as e_github:
            print(f"GitHub fallback failed: {e_github}")
            print("Generating synthetic classification dataset as a final fallback...")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=7, n_informative=5, n_classes=2, random_state=42)
            df = pd.DataFrame(X, columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])
            df['pclass'] = pd.qcut(df['pclass'], 3, labels=[1, 2, 3]).astype(int)
            df['sex'] = np.where(df['sex'] > 0, 'female', 'male')
            df['embarked'] = pd.qcut(df['embarked'], 3, labels=['S', 'C', 'Q']).astype(str)
            df['age'] = np.abs(df['age'] * 20 + 20).round(1)
            df['fare'] = np.abs(df['fare'] * 50 + 10).round(2)
            df['sibsp'] = np.clip(np.abs(df['sibsp']).astype(int), 0, 8)
            df['parch'] = np.clip(np.abs(df['parch']).astype(int), 0, 6)
            df['survived'] = y
            print("Synthetic dataset generated successfully!")
            return df


def main():
    # 1. Load Data
    df = load_data()
    if 'survived' in df.columns:
        df['survived'] = df['survived'].astype(int)

    print(f"\nDataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())

    # 2. Select Features and Target
    selected_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    available_features = [col for col in selected_features if col in df.columns]
    
    X = df[available_features]
    y = df['survived']

    # 3. Train-Test Split (Prevent leakage: split before pipeline fitting)
    print("\n--- 2. Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 4. Preprocessing Pipeline
    num_cols = [col for col in ['age', 'fare', 'sibsp', 'parch'] if col in X_train.columns]
    cat_cols = [col for col in ['sex', 'embarked', 'pclass'] if col in X_train.columns]

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

    # 5. Train Baseline Model
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
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Baseline Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'baseline_confusion_matrix.png'), dpi=300)
    plt.close()

    # 6. Hyperparameter Tuning using Cross-Validation
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

    # 7. Evaluate Tuned Model
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
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Tuned Random Forest Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'tuned_confusion_matrix.png'), dpi=300)
    plt.close()

    # 8. Compare Models
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
    plt.close()

    # 9. Feature Importance Analysis
    print("\n--- 8. Feature Importance Analysis ---")
    importances = tuned_rf.feature_importances_
    feat_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feat_importance_df.to_string(index=False))

    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_importance_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Importance Analysis (Tuned Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importances.png'), dpi=300)
    plt.close()
    print("\nAll plots have been saved to the workspace.")


if __name__ == '__main__':
    main()
