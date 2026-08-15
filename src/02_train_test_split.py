import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    print("=========================================")
    print("STEP 2: Train-Test Split (Prevent Leakage)")
    print("=========================================")
    
    # Load raw data
    raw_path = os.path.join(DATA_DIR, "titanic_raw.csv")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Error: {raw_path} not found. Please run 01_data_loading.py first!")
        return

    # Define features and target
    selected_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    available_features = [col for col in selected_features if col in df.columns]
    
    # Keep only selected columns plus the target
    cols_to_keep = available_features + ['survived']
    df_filtered = df[cols_to_keep]

    X = df_filtered[available_features]
    y = df_filtered['survived']

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Features selected for pipeline: {available_features}")
    print(f"Training split shape: {X_train.shape[0]} rows")
    print(f"Testing split shape: {X_test.shape[0]} rows")

    # Recombine features & target to save as partition CSVs
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(DATA_DIR, "titanic_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "titanic_test.csv"), index=False)
    print("\nPartitions successfully saved under 'data/' directory.")

if __name__ == '__main__':
    main()
