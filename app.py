import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# 1. LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# ==========================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("onehot_encoder.pkl")
oe = joblib.load("ordinal_encoder.pkl")
le = joblib.load("label_encoder.pkl")


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)


# ==========================================================
# 3. TITLE
# ==========================================================

st.title("💳 Credit Risk Prediction")
st.write(
    "Enter the applicant details to predict the credit risk."
)


# ==========================================================
# 4. USER INPUT
# ==========================================================

st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=1,
        value=50000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=60.0,
        value=5.0
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500,
        max_value=35000,
        value=10000
    )

with col2:

    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=5.0,
        max_value=25.0,
        value=10.0
    )

    loan_percent_income = st.number_input(
        "Loan Percent of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=30,
        value=5
    )


# ==========================================================
# 5. CATEGORICAL INPUTS
# ==========================================================

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "MORTGAGE", "OWN", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
        "EDUCATION",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

cb_person_default_on_file = st.selectbox(
    "Previous Default on File",
    ["N", "Y"]
)


# ==========================================================
# 6. PREDICTION BUTTON
# ==========================================================

if st.button("Predict Credit Risk"):

    # ------------------------------------------------------
    # BASIC VALIDATION
    # ------------------------------------------------------

    if cb_person_cred_hist_length > person_age:
        st.error(
            "Credit history length cannot be greater than age."
        )
        st.stop()

    # ------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------

    credit_history_age_ratio = (
        cb_person_cred_hist_length / person_age
    )

    # ------------------------------------------------------
    # NUMERICAL DATA
    # ------------------------------------------------------

    numerical_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length":
            [cb_person_cred_hist_length],
        "credit_history_age_ratio":
            [credit_history_age_ratio]
    })

    # ------------------------------------------------------
    # STANDARDIZATION
    # ------------------------------------------------------

    numerical_scaled = scaler.transform(
        numerical_data
    )

    numerical_scaled = pd.DataFrame(
        numerical_scaled,
        columns=numerical_data.columns
    )

    # ------------------------------------------------------
    # ONE-HOT ENCODING
    # ------------------------------------------------------

    categorical_data = pd.DataFrame({
        "person_home_ownership":
            [person_home_ownership],

        "loan_intent":
            [loan_intent]
    })

    ohe_data = ohe.transform(
        categorical_data
    )

    ohe_data = pd.DataFrame(
        ohe_data,
        columns=ohe.get_feature_names_out(
            ["person_home_ownership", "loan_intent"]
        )
    )

    # ------------------------------------------------------
    # ORDINAL ENCODING
    # ------------------------------------------------------

    grade_data = pd.DataFrame({
        "loan_grade": [loan_grade]
    })

    grade_data = oe.transform(
        grade_data
    )

    grade_data = pd.DataFrame(
        grade_data,
        columns=["loan_grade"]
    )

    # ------------------------------------------------------
    # LABEL ENCODING
    # ------------------------------------------------------

    default_data = le.transform(
        [cb_person_default_on_file]
    )

    default_data = pd.DataFrame(
        default_data,
        columns=["cb_person_default_on_file"]
    )

    # ------------------------------------------------------
    # COMBINE ALL FEATURES
    # ------------------------------------------------------

    final_input = pd.concat(
        [
            numerical_scaled,
            ohe_data,
            grade_data,
            default_data
        ],
        axis=1
    )

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(
        final_input
    )[0]

    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Higher Credit Risk - Likely Default"
        )

    else:

        st.success(
            "✅ Lower Credit Risk - Likely Non-Default"
        )

    # ------------------------------------------------------
    # PROBABILITY
    # ------------------------------------------------------

    probability = model.predict_proba(
        final_input
    )[0]

    st.write(
        f"Default Probability: "
        f"**{probability[1]:.2%}**"
    )

    st.write(
        f"Non-Default Probability: "
        f"**{probability[0]:.2%}**"
    )