import os
import pandas as pd

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_data():
    print("Loading Customer Churn datasets...")
    train_path = os.path.join(DATA_DIR, "churn-bigml-80.csv")
    test_path = os.path.join(DATA_DIR, "churn-bigml-20.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Train and Test datasets loaded successfully from CSV files!")
        # Combine them to represent the raw dataset
        combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        return combined_df
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        raise e

def main():
    print("=========================================")
    print("STEP 1: Data Loading & Initial Exploration")
    print("=========================================")
    df = load_data()
    
    print(f"\nCombined Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nData Types & Info:")
    df.info()
    
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    # Save raw combined data to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "churn_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"\nRaw dataset successfully saved to: {output_path}")

if __name__ == '__main__':
    main()
