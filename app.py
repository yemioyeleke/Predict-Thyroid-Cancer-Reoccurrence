import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load trained model and training columns
# ------------------------------------------------------------
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

try:
    with open("x_train_columns.pkl", "rb") as f:
        x_train_columns = pickle.load(f)
except FileNotFoundError:
    x_train_columns = []
    st.warning("⚠️ 'x_train_columns.pkl' not found. Ensure column names match your training data.")

st.set_page_config(page_title="Thyroid Recurrence Predictor", layout="wide")

st.title("🧠 Thyroid Cancer Recurrence Prediction App")
st.write("This app predicts the **likelihood of thyroid cancer recurrence** using machine learning model.")

# ------------------------------------------------------------
# Input fields
# ------------------------------------------------------------

st.subheader("Enter Patient Information")

# Numerical inputs
age = st.number_input("Age", min_value=0, max_value=120, value=45)
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, max_value=200.0, value=15.0)
lymph_nodes = st.number_input("Number of Lymph Nodes", min_value=0, max_value=50, value=2)

# Categorical inputs
thyroid_function = st.selectbox("Thyroid Function", ["Normal", "Abnormal"])
physical_exam = st.selectbox("Physical Examination", ["Benign", "Suspicious"])
adenopathy = st.selectbox("Adenopathy", ["Yes", "No"])
pathology = st.selectbox("Pathology", ["Papillary", "Follicular", "Medullary", "Anaplastic"])
focality = st.selectbox("Focality", ["Unifocal", "Multifocal"])
risk = st.selectbox("Risk Level", ["Low", "Intermediate", "High"])
t = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
n = st.selectbox("N Stage", ["N0", "N1"])
m = st.selectbox("M Stage", ["M0", "M1"])
stage = st.selectbox("Overall Stage", ["I", "II", "III", "IV"])
response = st.selectbox("Response to Treatment", ["Excellent", "Incomplete", "Indeterminate"])

# ------------------------------------------------------------
# Encoding based on training logic
# ------------------------------------------------------------

features = {}

# Add numeric variables
features["Age"] = age
features["Tumor_Size"] = tumor_size
features["Lymph_Nodes"] = lymph_nodes

# One-hot encoding logic (custom, same as training)
for col in x_train_columns:
    if col.startswith('Thyroid Function_'):
        features[col] = 1 if col.split('_')[-1] in thyroid_function else 0
    elif col.startswith('Physical Examination_'):
        features[col] = 1 if col.split('_')[-1] in physical_exam else 0
    elif col.startswith('Adenopathy_'):
        features[col] = 1 if col.split('_')[-1] in adenopathy else 0
    elif col.startswith('Pathology_'):
        features[col] = 1 if col.split('_')[-1] in pathology else 0
    elif col.startswith('Focality_'):
        features[col] = 1 if col.split('_')[-1] in focality else 0
    elif col.startswith('Risk_'):
        features[col] = 1 if col.split('_')[-1] in risk else 0
    elif col.startswith('T_'):
        features[col] = 1 if col.split('_')[-1] in t else 0
    elif col.startswith('N_'):
        features[col] = 1 if col.split('_')[-1] in n else 0
    elif col.startswith('M_'):
        features[col] = 1 if col.split('_')[-1] in m else 0
    elif col.startswith('Stage_'):
        features[col] = 1 if col.split('_')[-1] in stage else 0
    elif col.startswith('Response_'):
        features[col] = 1 if col.split('_')[-1] in response else 0
    elif col not in features:  # Ensure all training columns are present
        features[col] = 0

# Convert to DataFrame
input_df = pd.DataFrame([features])

# ------------------------------------------------------------
# Make prediction
# ------------------------------------------------------------

if st.button("Predict Recurrence Probability"):
    try:
        prediction_proba = xgb_model.predict_proba(input_df)[0][1]
        st.success(f"🩺 Predicted Probability of Recurrence: {prediction_proba:.2f}")

        if prediction_proba > 0.5:
            st.error("⚠️ High Risk of Recurrence")
        else:
            st.success("✅ Low Risk of Recurrence")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Developed by DataLab Analytics | Powered by XGBoost + Streamlit")
