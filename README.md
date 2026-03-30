# Credits-risk

## 1. Project Overview

This project focuses on **credit risk classification**, with the goal of predicting whether a loan is likely to become **high risk** or **non-risk** based on borrower and loan-related features. The project applies a full data mining and machine learning workflow, including data preprocessing, exploratory data analysis, hidden pattern discovery, feature selection, model training, and evaluation.

The main predictive model used in this project is **XGBoost (Extreme Gradient Boosting)**, which is well suited for structured tabular data and binary classification tasks.

---

## 2. Problem Statement

In credit lending, financial institutions need to assess whether a borrower is likely to repay a loan or default. Incorrect decisions may lead to financial loss, inefficient lending policies, and increased operational risk.

This project addresses the following problem:

> Given historical borrower and loan information, can we classify loan applications into different credit risk outcomes?

The target variable is based on **loan status**, which is transformed into a binary classification label:

* **0 = Fully Paid**
* **1 = Charged Off**

---

## 3. Objectives

The main objectives of this project are:

* To preprocess and clean a real-world credit risk dataset.
* To analyze important borrower and loan characteristics.
* To discover hidden patterns in the data through exploratory analysis and clustering.
* To build a machine learning model for credit risk classification.
* To evaluate model performance using standard classification metrics.
* To identify the factors that most strongly influence credit risk prediction.

---

## 4. Data Source

The dataset used in this project is based on historical loan records.

**Source:** LendingClub dataset / accepted loan records "https://www.kaggle.com/datasets/wordsforthewise/lending-club/data"
**Example file used:** `accepted_2007_to_2018Q4.csv`

This dataset contains information related to:

* borrower profile
* financial capacity
* credit history
* loan characteristics
* repayment outcome

**Target field:** `loan_status`

Only the following two classes are retained for binary classification:

* `Fully Paid`
* `Charged Off`

Then the label is created as:

```python
label = {
    "Fully Paid": 0,
    "Charged Off": 1
}
```

> Note: If you have an official source link for the dataset, replace this section with the exact URL or citation.

---

## 5. Selected Features

The project selects variables based on the **5Cs of Credit** framework:

### 5.1 Character

* `pub_rec`
* `num_tl_90g_dpd_24m`

### 5.2 Capacity

* `annual_inc`
* `dti`
* `loan_amnt`
* `term`
* `installment`

### 5.3 Capital

* `tot_cur_bal`
* `open_acc`
* `total_acc`
* `revol_util`
* `inq_last_6mths`
* `fico_range_low`
* `fico_range_high`

### 5.4 Collateral

* `home_ownership`

### 5.5 Conditions

* `purpose`
* `verification_status`

### 5.6 Target

* `label`

These features were selected because they represent important aspects of borrower risk, repayment ability, and financial stability.

---

## 6. Project Workflow

The project follows the workflow below:

1. **Data Collection**
   Load the raw credit risk dataset.

2. **Data Preprocessing**

   * retain only relevant loan statuses
   * create binary label
   * select important features
   * handle missing values
   * encode categorical variables
   * scale or transform numeric variables if needed

3. **Exploratory Data Analysis (EDA)**

   * distribution analysis
   * boxplots and histograms
   * correlation heatmap
   * class distribution analysis

4. **Hidden Pattern Discovery**

   * clustering
   * cluster profiling
   * heatmap analysis
   * interpretation of risk segments

5. **Model Building**
   Train an **XGBoost classifier** for binary classification.

6. **Model Evaluation**
   Evaluate the model using classification metrics.

7. **Interpretation and Insight Extraction**
   Identify important features and explain how they affect credit risk.

---

## 7. Data Preprocessing

The preprocessing stage includes the following steps:

### 7.1 Filtering Target Classes

Only two loan outcomes are used:

* `Fully Paid`
* `Charged Off`

### 7.2 Creating Binary Label

```python
df["label"] = df["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1
})
```

### 7.3 Feature Selection

Only relevant variables are retained based on domain knowledge and the 5Cs of Credit.

### 7.4 Missing Value Handling

Missing values are removed or treated depending on the preprocessing strategy.

Example:

```python
df_final = df[selected_columns].dropna()
```

### 7.5 Encoding Categorical Features

Categorical variables such as `home_ownership`, `purpose`, and `verification_status` are encoded before model training.

### 7.6 Train-Test Split

The cleaned dataset is split into training and testing subsets for model development and evaluation.

---

## 8. Model

The main model used in this project is:

## XGBoost Classifier

**XGBoost** is an efficient gradient boosting algorithm that performs well on tabular structured data. It is widely used in classification problems because of its ability to:

* handle non-linear relationships
* capture feature interactions
* manage imbalanced data more effectively than many baseline models
* provide feature importance for interpretation

### Why XGBoost?

XGBoost was chosen because:

* it is powerful for binary classification tasks
* it performs well on large tabular datasets
* it is robust to noisy features
* it supports regularization to reduce overfitting

---

## 9. Evaluation Metrics

The model can be evaluated using the following metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **ROC-AUC**
* **Confusion Matrix**

These metrics help assess not only the overall correctness of the model but also its ability to identify risky borrowers.

### Example Interpretation

* **Accuracy** measures the overall proportion of correct predictions.
* **Precision** measures how many predicted risky loans are truly risky.
* **Recall** measures how many truly risky loans are successfully detected.
* **F1-score** balances precision and recall.
* **ROC-AUC** evaluates the model’s ability to distinguish between classes.

---

## 10. Hidden Pattern Discovery

In addition to predictive modeling, this project also explores **hidden patterns** in the credit data.

This part aims to identify meaningful customer groups such as:

* younger borrowers with lower income and smaller loans
* borrowers with high debt-to-income ratio
* financially stable borrowers with strong repayment ability
* groups with elevated default risk patterns

Possible techniques include:

* **K-Means Clustering**
* **Hierarchical Clustering**
* **Heatmap-based analysis**
* **Cluster profiling**

This analysis helps explain the data beyond pure prediction and supports better business insight generation.

---

## 11. Expected Outputs

The main outputs of the project include:

* a cleaned and processed dataset
* exploratory visualizations
* hidden pattern analysis results
* trained XGBoost classification model
* evaluation results
* interpretation of key risk factors

---

## 12. Project Structure

A suggested project structure is shown below:

```text
Credits-risk/
│
├── data/
│   ├── accepted_2007_to_2018Q4.csv
│   └── credit_risk_dataset_5c.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   ├── hidden_pattern.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── train_xgboost.py
│   └── evaluate.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── requirements.txt
└── README.md
```

---

## 13. How to Run the Project

### 13.1 Clone the Repository

```bash
git clone https://github.com/Tuan24455/Credits-risk.git
cd Credits-risk
```

### 13.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 13.3 Run the Project

Example:

```bash
python src/train_xgboost.py
```

Or open notebooks:

```bash
jupyter notebook
```

---

## 14. Technologies and Libraries

This project may use the following tools and libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SciPy
* Jupyter Notebook

---

## 15. Business Value

This project provides value in several ways:

* improves credit risk assessment
* supports better lending decisions
* reduces the chance of approving high-risk borrowers
* helps discover hidden borrower segments
* supports data-driven financial decision making

---

## 16. Future Improvements

Possible future improvements include:

* hyperparameter tuning for XGBoost
* class imbalance handling using SMOTE or weighted learning
* comparing XGBoost with other models such as Random Forest, Logistic Regression, LightGBM, or CatBoost
* model explainability using SHAP
* deploying the model as a web application or dashboard

---

## 17. Author

**Author:** Tuan24455
**Repository:** Credits-risk
