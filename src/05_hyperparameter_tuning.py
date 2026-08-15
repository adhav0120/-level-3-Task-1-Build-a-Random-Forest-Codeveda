import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def main():
    print("=========================================")
    print("STEP 5: Hyperparameter Tuning (Grid Search)")
    print("=========================================")
    
    # Define paths relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Load training arrays
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train_preprocessed.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    except FileNotFoundError:
        print("Error: Preprocessed data files not found. Please run 03_preprocessing.py first!")
        return

    # Setup parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Grid Search with 5-fold CV
    print("Running GridSearchCV with 5-fold cross-validation on F1-score...")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    print("\nBest Hyperparameters Found:")
    for param, val in grid_search.best_params_.items():
        print(f" - {param}: {val}")
        
    print(f"Best 5-Fold Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

    # Save best parameters to JSON
    os.makedirs(MODELS_DIR, exist_ok=True)
    params_path = os.path.join(MODELS_DIR, "best_hyperparameters.json")
    with open(params_path, "w") as f:
        json.dump(grid_search.best_params_, f)
        
    print(f"\nBest hyperparameters successfully written to: {params_path}")

if __name__ == '__main__':
    main()
