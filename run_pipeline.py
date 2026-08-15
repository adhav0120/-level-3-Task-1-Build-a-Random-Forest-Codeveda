import os
import subprocess
import sys

# Get root directory where run_pipeline.py resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    print(f"\n>>> Running: {script_name}...")
    # Get absolute path to the script in src/
    script_path = os.path.join(BASE_DIR, "src", script_name)
    # Run the script using the current python executable
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with exit code {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)
    print(f">>> Completed: {script_name} successfully.\n")

def main():
    pipeline_scripts = [
        "01_data_loading.py",
        "02_train_test_split.py",
        "03_preprocessing.py",
        "04_baseline_model.py",
        "05_hyperparameter_tuning.py",
        "06_model_evaluation.py",
        "07_feature_importance.py"
    ]
    
    print("=========================================")
    print("STARTING MODULAR MACHINE LEARNING PIPELINE")
    print("=========================================")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print("=========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=========================================")

if __name__ == '__main__':
    main()
