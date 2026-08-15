import os
import json
import joblib
import numpy as np
import pandas as pd

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    print("=========================================")
    print("STEP 3: Preprocessing Pipeline Construction")
    print("=========================================")
    
    # Load partition data
    train_path = os.path.join(DATA_DIR, "titanic_train.csv")
    test_path = os.path.join(DATA_DIR, "titanic_test.csv")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Train/Test files not found. Please run 02_train_test_split.py first!")
        return

    # Separate target
    target = 'survived'
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Define columns
    num_cols = [col for col in ['age', 'fare', 'sibsp', 'parch'] if col in X_train.columns]
    cat_cols = [col for col in ['sex', 'embarked', 'pclass'] if col in X_train.columns]

    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")

    # Numerical Transformer
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Fit and transform
    print("\nFitting preprocessing steps on training data and transforming datasets...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Extract feature names
    num_feature_names = num_cols
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_feature_names + cat_feature_names

    # Save outputs
    print("\nSaving preprocessed data and models...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "X_train_preprocessed.npy"), X_train_preprocessed)
    np.save(os.path.join(DATA_DIR, "X_test_preprocessed.npy"), X_test_preprocessed)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test.to_numpy())

    # Save fitted preprocessor
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    
    # Save feature names list
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    print("Preprocessed feature count:", len(feature_names))
    print("Preprocessed feature names:", feature_names)
    print("\nPreprocessing complete. Serialized objects saved.")

if __name__ == '__main__':
    main()
