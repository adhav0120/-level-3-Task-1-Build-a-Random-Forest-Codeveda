# Customer Churn Prediction: End-to-End Random Forest Classification Pipeline

An end-to-end Machine Learning pipeline utilizing a **Random Forest Classifier** to predict customer churn. The project features clean preprocessing pipelines, hyperparameter tuning with 5-fold cross-validation, feature importance analysis, and a modular enterprise-style design.

---

## 🎯 Objective
Develop, optimize, and evaluate a Random Forest binary classification pipeline to predict customer churn, comparing a baseline default model against an optimized, hyperparameter-tuned model.

## 📝 Problem Statement
Given customer subscription, plan, and usage statistics, classify whether a customer will churn (`1`) or not churn (`0`). This is an essential business intelligence classification task, where we seek to build a model that generalizes well to unseen test data, minimizing both False Positives (predicting churn when they won't, leading to wasted retention costs) and False Negatives (missing a churning customer, leading to lost revenue).

## 📊 Dataset Description
The dataset contains customer telecom usage statistics. We use pre-split partitions:
- **Training Set (`churn-bigml-80.csv`)**: 2,668 rows
- **Testing Set (`churn-bigml-20.csv`)**: 665 rows

The features include:
- **`State`**: The US state (categorical/high cardinality).
- **`Account length`**: Length of the customer account in months (numerical).
- **`Area code`**: Area code of the customer (categorical).
- **`International plan`**: Whether the customer has an international plan (Yes/No categorical).
- **`Voice mail plan`**: Whether the customer has a voicemail plan (Yes/No categorical).
- **`Number vmail messages`**: Number of voicemail messages (numerical).
- **`Total day minutes` / `Total day calls` / `Total day charge`**: Usage stats during the day (numerical).
- **`Total eve minutes` / `Total eve calls` / `Total eve charge`**: Usage stats during the evening (numerical).
- **`Total night minutes` / `Total night calls` / `Total night charge`**: Usage stats during the night (numerical).
- **`Total intl minutes` / `Total intl calls` / `Total intl charge`**: Usage stats for international calls (numerical).
- **`Customer service calls`**: Number of calls made to customer service (numerical).
- **`Churn` (Target)**: Customer churn status (True = Churned, False = Retained).

## 🛠️ Technologies Used
*   **Programming Language**: Python 3.10
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn
*   **Visualization**: Matplotlib, Seaborn
*   **Model Serialization**: Joblib

---

## 📂 Project Structure
The repository is structured both for interactive analysis and modular, step-by-step pipeline execution:

```
├── .venv/                           # Python Virtual Environment
├── data/                            # Datasets & intermediate preprocessed numpy arrays
│   ├── churn-bigml-80.csv           # Original training set
│   ├── churn-bigml-20.csv           # Original testing set
│   ├── churn_raw.csv                # Combined raw dataset
│   ├── churn_train.csv              # Processed train set
│   ├── churn_test.csv               # Processed test set
│   ├── X_train_preprocessed.npy
│   ├── X_test_preprocessed.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── models/                          # Serialized preprocessors, models, and JSON metrics
│   ├── preprocessor.joblib
│   ├── tuned_model.joblib
│   ├── best_hyperparameters.json
│   ├── baseline_metrics.json
│   ├── tuned_metrics.json
│   └── feature_names.json
│
├── notebooks/                       # Interactive Jupyter Notebooks
│   └── random_forest_classification.ipynb
│
├── plots/                           # Generated visualization PNG files
│   ├── baseline_confusion_matrix.png
│   ├── tuned_confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importances.png
│
├── src/                             # Python source code
│   ├── 01_data_loading.py
│   ├── 02_train_test_split.py
│   ├── 03_preprocessing.py
│   ├── 04_baseline_model.py
│   ├── 05_hyperparameter_tuning.py
│   ├── 06_model_evaluation.py
│   ├── 07_feature_importance.py
│   └── random_forest_classification.py
│
├── requirements.txt                 # Frozen project dependencies
├── README.md                        # Project documentation (this file)
└── run_pipeline.py                  # Master pipeline runner (executes steps 1-7 in src/)
```

---

## 🚀 Installation Instructions
1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up a Virtual Environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment (Windows Powershell)
   .venv\Scripts\Activate.ps1

   # Activate virtual environment (macOS/Linux)
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run
This project offers three flexible execution modes depending on your workflow:

### Option A: Run the Master Pipeline (Orchestrator)
Orchestrates the entire modular machine learning pipeline sequentially:
```bash
python run_pipeline.py
```

### Option B: Run the Monolithic Script
Executes the entire end-to-end process from a single standalone script:
```bash
python src/random_forest_classification.py
```

### Option C: Run Interactively
Launch Jupyter Notebook to inspect data, intermediate steps, and plots inline:
```bash
cd notebooks
jupyter notebook random_forest_classification.ipynb
```

---

## 🔬 Methodology

### Preprocessing
To guarantee rigorous model evaluation and prevent **data leakage**, the preprocessing pipeline adheres to the following rules:
*   The raw data split is pre-established (80% train, 20% test).
*   A Scikit-Learn `ColumnTransformer` handles different data types separately:
    *   **Numerical Features**: Missing values are imputed using the **median**, followed by standard normalization via **`StandardScaler`**.
    *   **Categorical Features**: Missing values are imputed using the **mode (most frequent)**, followed by binary encoding via **`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`**.
*   The transformer is fitted exclusively on the training set and applied to both splits. The preprocessor configuration is serialized to `models/preprocessor.joblib`.

### Model Development
Two models are built and compared:
1.  **Baseline Classifier**: A standard `RandomForestClassifier` with default scikit-learn hyperparameters.
2.  **Tuned Classifier**: An optimized Random Forest model using `GridSearchCV` with **5-fold cross-validation** to search the parameter space. It optimizes for **F1-score** across these configurations:
    *   `n_estimators`: `[50, 100, 200]`
    *   `max_depth`: `[None, 5, 10, 15]`
    *   `min_samples_split`: `[2, 5, 10]`
    *   `min_samples_leaf`: `[1, 2, 4]`
    *   `max_features`: `['sqrt', 'log2', None]`

The optimal hyperparameters are saved in `models/best_hyperparameters.json` and the final trained model is stored as `models/tuned_model.joblib`.

### Evaluation Metrics
Both models are evaluated on the unseen test set using standard binary classification metrics:
*   **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ (Overall correctness)
*   **Precision**: $\frac{TP}{TP + FP}$ (Minimizes False Positives)
*   **Recall**: $\frac{TP}{TP + FN}$ (Minimizes False Negatives)
*   **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic mean balancing precision and recall)
*   **Confusion Matrix**: Visual representation of classification errors.
