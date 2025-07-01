# Loan Approval Prediction 

## Overview

This project predicts loan approval using machine learning models trained on a dataset of applicant details. It aims to:
- Enhance decision-making in loan processing.
- Promote transparency through explainable AI techniques like SHAP.
- Provide a Streamlit web app for real-time, user-friendly predictions.

👉 [**Access the Application**](https://loanapprover.streamlit.app/)

## Dataset and Variables

The project uses the Loan Prediction Dataset from Kaggle, containing features across demographic, socioeconomic, and financial categories:

| **Feature Type** | **Features**                                         | **Description**                         |
|------------------|-----------------------------------------------------|-----------------------------------------|
| Demographic      | `Gender`, `Married`, `Dependents`                   | Applicant details.                      |
| Socioeconomic    | `Education`, `Self_Employed`, `ApplicantIncome`     | Income and education info.              |
| Loan Details     | `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`  | Loan and credit details.                |
| Target Variable  | `Loan_Status`                                       | Approved (`Y`) or Rejected (`N`).       |


## Data Analysis Workflow

- **Data Preprocessing:** Handling missing values, outliers, and skewed distributions with transformations.
- **Feature Engineering:** Creation of new features based on domain knowledge to improve model performance.
- **Data Visualization:** Visual exploration of key variables and relationships to understand data distributions and correlations.
- **Model Training and Comparison:**
  - Evaluated models: KNN, SVM, Random Forest, and XGBoost.
  - Random Forest was selected for its performance (highest F1-score) and interpretability.
  - Feature Importance Analysis
- **Bias Reduction:** Removed demographic features (e.g., gender, marital status) to ensure fairness and focus on financial predictors.
- **Model Explanation:** Used SHAP values to provide interpretable insights into individual predictions and clearly outline model decisions for transparency

## Data Insights

- **Top Predictors:** Credit history, total income, balance income, and EMI were identified as the most impactful features for loan approval.
- **Ethical Considerations:** Exclude demographic features with low predictive power and potential for discrimination.

## Model Deployment and Usage

- **Saving the Model:** The trained Random Forest classifier is saved using the `joblib` library.
- **Streamlit App:** A web application is developed using `streamlit` to create an interactive interface for users to input loan application details.
   
---

## Steps to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/gianlucasposito/Loan-Approval-Prediction
    cd Loan-Approval-Prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Contributions

Contributions are welcome! Please open an issue or submit a pull request for enhancements or bug fixes.

