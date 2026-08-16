# 🎯 Candidate Selection Prediction Model

A Machine Learning classification project that predicts whether a job candidate is likely to be **selected or not selected** based on candidate-related academic, technical, professional, and skill-based attributes.

The project implements a complete Machine Learning pipeline including **data ingestion, data preprocessing, categorical encoding, missing-value handling, feature scaling, class balancing using SMOTE, model training, evaluation, and model serialization**.

---

## 📌 Project Overview

Recruitment teams often need to evaluate a large number of candidates based on multiple factors such as education, experience, technical skills, internships, projects, certifications, and other professional attributes.

This project uses Machine Learning to automate the candidate selection prediction process. The model takes candidate information as input and predicts:

- **1 → Selected**
- **0 → Not Selected**

The project is structured as a modular Machine Learning pipeline so that each stage of the workflow can be maintained and reused independently.

---

## 🚀 Features

- 📥 Automated data ingestion using Pandas
- 🧹 Data preprocessing and cleaning
- 🔄 Duplicate record handling
- 🗑️ Removal of unnecessary features
- 🔤 Categorical feature encoding
- 🎯 Target variable encoding
- 🩹 Missing-value handling
- 📊 Numerical feature scaling
- 🔀 Train-test data splitting
- ⚖️ Class imbalance handling using SMOTE
- 🌲 Random Forest classification
- 📈 Model evaluation using accuracy and classification report
- 💾 Trained model saved as a `.pkl` file
- 🧩 Modular project structure

---

## 📁 Dataset

The project uses `resume_screening_dataset.csv`, containing candidate-related information used to determine whether a candidate is selected.

| Column                     | Description                                                    |
| --------------------------- | ---------------------------------------------------------------- |
| `age`                      | Age of the candidate                                            |
| `gender`                   | Gender of the candidate                                         |
| `degree`                   | Highest academic degree held by the candidate                   |
| `specialization`           | Field of study / academic specialization                        |
| `university_tier`          | Tier/ranking category of the candidate's university              |
| `cgpa`                     | Cumulative Grade Point Average                                  |
| `years_experience`         | Total years of professional work experience                     |
| `internships_count`        | Number of internships completed                                 |
| `projects_count`           | Number of projects completed                                    |
| `certifications_count`     | Number of professional certifications earned                    |
| `python_skill`             | Proficiency level in Python                                     |
| `sql_skill`                | Proficiency level in SQL                                        |
| `machine_learning_skill`   | Proficiency level in Machine Learning                           |
| `deep_learning_skill`      | Proficiency level in Deep Learning                               |
| `communication_skill`      | Proficiency level in communication                               |
| `leadership_skill`         | Proficiency level in leadership                                 |
| `github_projects`          | Number of projects published on GitHub                          |
| `kaggle_activity_score`    | Score representing activity/engagement on Kaggle                |
| `hackathons_participated`  | Number of hackathons the candidate has participated in          |
| `expected_salary_lpa`      | Candidate's expected salary (in Lakhs Per Annum)                 |
| `employment_gap_months`    | Number of months of employment gap                              |
| `selected` *(target)*      | Whether the candidate was selected (`Yes` → 1, `No` → 0)        |

The target variable is `selected`, encoded as:

```
Yes → 1
No  → 0
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Programming language                 |
| Pandas           | Data loading and manipulation        |
| NumPy            | Numerical operations                 |
| Scikit-learn     | Machine Learning and preprocessing   |
| Imbalanced-learn | SMOTE class balancing                |
| Matplotlib       | Data visualization                   |
| Seaborn          | Data visualization                   |
| FLAML            | Automated Machine Learning support   |
| Pickle           | Model serialization                  |
| UV               | Python project/dependency management |

The project dependencies are defined in `pyproject.toml`.

---

## 📂 Project Structure

```
Candidate_Selection_Prediction_Model/
│
├── data/
│   └── resume_screening_dataset.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   └── candidateselectionpredictionmodel/
│       ├── __init__.py
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       └── model_building.py
│
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

The repository follows a modular structure separating **data ingestion, preprocessing, and model building**.

---

## 🔄 Machine Learning Workflow

```
┌─────────────────────┐
│     Raw Dataset      │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│   Data Ingestion      │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ Data Preprocessing    │
│                       │
│ • Remove duplicates   │
│ • Drop unnecessary    │
│   columns             │
│ • Encode target       │
│ • Handle missing      │
│   values              │
│ • Encode categories   │
│ • Scale numerical     │
│   features            │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│   Train/Test Split    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│       SMOTE           │
│ Class Balancing       │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ Random Forest         │
│    Classifier         │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│    Evaluation          │
│                       │
│ • Accuracy             │
│ • Classification       │
│   Report               │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│    model.pkl           │
│ Saved ML Model         │
└─────────────────────┘
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline performs the following operations:

### 1. Remove Duplicate Records
Duplicate records are identified and handled before model training.

### 2. Remove Unnecessary Columns
The following columns are removed from the dataset:
```
candidate_id
location
```

### 3. Encode Target Variable
The `selected` target column is converted into numerical values:
```
No  → 0
Yes → 1
```

### 4. Separate Features and Target
```
X = Input Features
y = selected
```

### 5. Identify Numerical and Categorical Features
Numerical and categorical columns are processed separately.

### 6. Handle Missing Values
For numerical features:
```
SimpleImputer(strategy="mean")
```
For categorical features:
```
SimpleImputer(strategy="most_frequent")
```

### 7. Encode Categorical Features
Categorical variables are transformed using:
```
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

### 8. Scale Numerical Features
Numerical features are normalized using:
```
MinMaxScaler()
```

### 9. Handle Class Imbalance
SMOTE (**Synthetic Minority Over-sampling Technique**) is applied to the training data to address class imbalance.

The preprocessing implementation combines these operations using Scikit-learn `Pipeline` and `ColumnTransformer`.

---

## 🤖 Machine Learning Model

### Random Forest Classifier
The project uses a **Random Forest Classifier** for candidate selection prediction.
```
RandomForestClassifier(random_state=1)
```
Random Forest is an ensemble learning algorithm that combines multiple decision trees to make a final prediction.

### Why Random Forest?
Random Forest is suitable for this problem because:
- It handles both complex and non-linear relationships.
- It works well with many input features.
- It is relatively robust against overfitting.
- It can capture interactions between candidate attributes.
- It performs well for classification problems.

The model is trained using the preprocessed and SMOTE-balanced training dataset.

---

## 📊 Model Evaluation

The trained Random Forest Classifier was evaluated using **Precision, Recall, F1-score, Support, and Accuracy**.

### Classification Report
```
              precision    recall  f1-score   support

           0       0.90      0.89      0.90      9072
           1       0.83      0.85      0.84      5928

    accuracy                           0.87     15000
```

### Model Accuracy
```
RandomForestClassifier Accuracy = 87.37%
```
**Overall Accuracy: 87.37%**

---

## 💾 Model Serialization

After training, the Random Forest model is saved using Python's `pickle` module:
```
models/model.pkl
```
This allows the trained model to be reused later without retraining it from scratch.

---

## ⚙️ Installation

### 1. Clone the Repository
```
git clone https://github.com/deveshdubey18/Candidate_Selection_Prediction_Model.git
```

### 2. Navigate to the Project
```
cd Candidate_Selection_Prediction_Model
```

### 3. Create a Virtual Environment
Using Python:
```
python -m venv .venv
```
Activate it on Windows:
```
.venv\Scripts\activate
```
On Linux/macOS:
```
source .venv/bin/activate
```

### 4. Install Dependencies
Using pip:
```
pip install -e .
```
Or, if you are using UV:
```
uv sync
```
The project is configured for Python **3.13 or higher**.

---

## ▶️ Run the Project

Run the main pipeline using:
```
python main.py
```

The `main.py` script executes the pipeline in the following order:
```
Data Ingestion
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving
```

---

## 📈 Expected Output

After running the project, the program displays the dataset shape and the classification report.

```
Classification report:

              precision    recall    f1-score    support

           0       ...
           1       ...

    accuracy                   ...
```

The final model performance is also returned in the following format:
```
RandomForestClassifier Accuracy = XX.XX%
```

The trained model is then stored in:
```
models/model.pkl
```

---

## 🧠 Key Machine Learning Concepts Used

This project demonstrates practical implementation of:
- Data Ingestion
- Data Cleaning
- Feature Engineering
- Target Encoding
- One-Hot Encoding
- Missing Value Imputation
- Feature Scaling
- Train-Test Split
- SMOTE
- Machine Learning Pipelines
- ColumnTransformer
- Random Forest Classification
- Model Evaluation
- Model Serialization

---

## 🔮 Future Improvements

- [ ] Add exploratory data analysis (EDA)
- [ ] Add feature importance visualization
- [ ] Compare multiple classification algorithms
- [ ] Perform hyperparameter tuning
- [ ] Add cross-validation
- [ ] Add ROC-AUC evaluation
- [ ] Add confusion matrix visualization
- [ ] Improve data ingestion using relative paths
- [ ] Create a prediction interface using Streamlit
- [ ] Deploy the model as a web application
- [ ] Add automated testing
- [ ] Add CI/CD using GitHub Actions

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

Candidate selection involves sensitive and complex human decisions. A Machine Learning prediction should not be treated as the sole basis for making hiring decisions. Real-world recruitment systems should include appropriate human oversight, fairness evaluation, privacy protections, and bias monitoring.

---

## 👨‍💻 Author

**Devesh Dubey**
GitHub: [@deveshdubey18](https://github.com/deveshdubey18)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is available for educational and personal use.
