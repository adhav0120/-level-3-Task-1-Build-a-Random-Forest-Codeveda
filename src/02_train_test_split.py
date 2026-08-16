import os
import pandas as pd

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    print("=========================================")
    print("STEP 2: Train-Test Split (Retention of Pre-split Data)")
    print("=========================================")
    
    # Load training and testing sets
    train_raw_path = os.path.join(DATA_DIR, "churn-bigml-80.csv")
    test_raw_path = os.path.join(DATA_DIR, "churn-bigml-20.csv")
    
    try:
        train_df = pd.read_csv(train_raw_path)
        test_df = pd.read_csv(test_raw_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the datasets are in the 'data/' folder.")
        return

    # Define features and target
    num_features = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes',
        'Total intl calls', 'Total intl charge', 'Customer service calls'
    ]
    cat_features = ['State', 'Area code', 'International plan', 'Voice mail plan']
    selected_features = num_features + cat_features
    target = 'Churn'
    
    # Check if all features exist
    available_features = [col for col in selected_features if col in train_df.columns]
    
    # Process train split
    X_train = train_df[available_features].copy()
    y_train = train_df[target].astype(int) # True/False to 1/0
    
    # Process test split
    X_test = test_df[available_features].copy()
    y_test = test_df[target].astype(int) # True/False to 1/0
    
    # Convert Area code to string to ensure categorical treatment
    if 'Area code' in X_train.columns:
        X_train['Area code'] = X_train['Area code'].astype(str)
        X_test['Area code'] = X_test['Area code'].astype(str)

    print(f"Features selected for pipeline: {available_features}")
    print(f"Training split shape: {X_train.shape[0]} rows")
    print(f"Testing split shape: {X_test.shape[0]} rows")

    # Recombine features & target to save as partition CSVs
    train_processed = pd.concat([X_train, y_train], axis=1)
    test_processed = pd.concat([X_test, y_test], axis=1)

    train_processed.to_csv(os.path.join(DATA_DIR, "churn_train.csv"), index=False)
    test_processed.to_csv(os.path.join(DATA_DIR, "churn_test.csv"), index=False)
    print("\nPartitions successfully saved under 'data/' directory.")

if __name__ == '__main__':
    main()
