# 💳 Credit Risk Prediction ML App

A Machine Learning web application that predicts the **credit risk of a loan applicant** using applicant financial and loan-related information.

## 🚀 Live Demo

👉 **Try the live application:**  
https://creditriskmlapp-euuai2dpvu7e9vcwbdjbnl.streamlit.app/

## 📌 Project Overview

This project uses Machine Learning to predict whether a loan applicant is likely to **default** or **not default** based on historical credit-risk data.

The project covers an end-to-end Machine Learning workflow:

**Data Cleaning → Feature Engineering → Encoding → Scaling → Model Training → Model Evaluation → Model Deployment**

## 🎯 Objective

The main objective is to build a Machine Learning model that can help identify potential loan default risk from applicant information such as income, loan amount, interest rate, employment length, loan grade, and credit history.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- VS Code

## 🤖 Machine Learning Model

**Logistic Regression** was used as the classification model.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 85.18% |
| Precision | 73.88% |
| Recall | 49.86% |
| F1 Score | 59.54% |

> The model's predictions are based on patterns learned from the training dataset and should not be considered a guarantee of actual loan repayment behaviour.

## 🔧 Feature Engineering

A new feature was created:

**Credit History Age Ratio**

```text
Credit History Length / Person Age
```

This feature represents the applicant's credit history length relative to their age.

## 🔤 Categorical Encoding

Different encoding techniques were selected based on the nature of the categorical variables:

- **One-Hot Encoding** → Home Ownership, Loan Intent
- **Ordinal Encoding** → Loan Grade
- **Label Encoding** → Previous Default on File

## 📊 Feature Scaling

**StandardScaler** was used for the numerical features.

The scaler fitted during model training is reused when processing new user inputs in the deployed application.

## 💾 Model Deployment

The trained Machine Learning model and preprocessing objects were saved using **Joblib**:

```text
model.pkl
scaler.pkl
onehot_encoder.pkl
ordinal_encoder.pkl
label_encoder.pkl
```

The application loads these saved objects and applies the same preprocessing workflow to new user inputs before generating a prediction.

## 🌐 Streamlit Application

The Streamlit application allows users to enter applicant details and receive:

- Credit risk prediction
- Default probability
- Non-default probability
- Basic input validation

### Application Workflow

```text
User Input
    ↓
Input Validation
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Logistic Regression
    ↓
Risk Prediction
```

## 📁 Project Structure

```text
Credit_Risk_ML_App/
│
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── onehot_encoder.pkl
├── ordinal_encoder.pkl
└── label_encoder.pkl
```

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest and other classification models
- Perform hyperparameter tuning
- Improve recall for default detection
- Add confusion matrix and ROC-AUC analysis
- Improve input validation and error handling
- Add a more advanced user interface

## 👨‍💻 Author

**Machapathri Vamshi**

B.Tech Information Technology Graduate | Data Science & Machine Learning

## 🔗 Links

**Live Application:**  
https://creditriskmlapp-euuai2dpvu7e9vcwbdjbnl.streamlit.app/

**GitHub Repository:**  
Add your repository link here.

---

⭐ If you find this project useful, consider giving the repository a star.
