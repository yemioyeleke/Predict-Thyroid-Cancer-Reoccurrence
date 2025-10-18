import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Load Trained Model
# -------------------------------
MODEL_PATH = 'xgb_model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please ensure 'xgb_model.pkl' is in the app directory.")
    st.stop()

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# -------------------------------
# Columns used during training
# -------------------------------
x_train_columns = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy',
    'Thyroid Function_Clinical Hyperthyroidism',
    'Thyroid Function_Clinical Hypothyroidism',
    'Thyroid Function_Euthyroid',
    'Thyroid Function_Subclinical Hyperthyroidism',
    'Thyroid Function_Subclinical Hypothyroidism',
    'Physical Examination_Diffuse goiter',
    'Physical Examination_Multinodular goiter',
    'Physical Examination_Normal',
    'Physical Examination_Single nodular goiter-left',
    'Physical Examination_Single nodular goiter-right',
    'Adenopathy_Bilateral', 'Adenopathy_Extensive', 'Adenopathy_Left',
    'Adenopathy_No', 'Adenopathy_Posterior', 'Adenopathy_Right',
    'Pathology_Follicular', 'Pathology_Hurthel cell',
    'Pathology_Micropapillary', 'Pathology_Papillary',
    'Focality_Multi-Focal', 'Focality_Uni-Focal', 'Risk_High',
    'Risk_Intermediate', 'Risk_Low', 'T_T1a', 'T_T1b', 'T_T2', 'T_T3a',
    'T_T3b', 'T_T4a', 'T_T4b', 'N_N0', 'N_N1a', 'N_N1b', 'M_M0', 'M_M1',
    'Stage_I', 'Stage_II', 'Stage_III', 'Stage_IVA', 'Stage_IVB',
    'Response_Biochemical Incomplete', 'Response_Excellent',
    'Response_Indeterminate', 'Response_Structural Incomplete'
]

# -------------------------------
# App Title and Sidebar
# -------------------------------
st.set_page_config(page_title="Thyroid Cancer Recurrence Prediction", layout="wide")
st.title("🧬 Thyroid Cancer Recurrence Prediction App")
st.markdown("""
**Developed by DataLab Team**

This app serves as a clinical decision support tool, enabling healthcare professionals 
to **predict the likelihood of thyroid cancer recurrence** based on patient clinical data.
""")

st.sidebar.title("About This App")
st.sidebar.info("""
Enter patient details using the fields below.
The model will estimate the **probability of cancer recurrence**.
""")

# -------------------------------
# User Input Function
# -------------------------------
def user_input_features():
    age = st.slider('Age', 0, 120, 30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    smoking = st.selectbox('Smoking', ['Yes', 'No'])
    hx_smoking = st.selectbox('Hx Smoking', ['Yes', 'No'])
    hx_radiotherapy = st.selectbox('Hx Radiotherapy', ['Yes', 'No'])

    thyroid_function = st.selectbox(
        'Thyroid Function',
        ['Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Euthyroid', 
         'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism']
    )
    physical_exam = st.selectbox(
        'Physical Examination',
        ['Diffuse goiter', 'Multinodular goiter', 'Normal', 
         'Single nodular goiter-left', 'Single nodular goiter-right']
    )
    adenopathy = st.selectbox(
        'Adenopathy',
        ['Bilateral', 'Extensive', 'Left', 'No', 'Posterior', 'Right']
    )
    pathology = st.selectbox(
        'Pathology',
        ['Follicular', 'Hurthel cell', 'Micropapillary', 'Papillary']
    )
    focality = st.selectbox('Focality', ['Multi-Focal', 'Uni-Focal'])
    risk = st.selectbox('Risk', ['High', 'Intermediate', 'Low'])
    t = st.selectbox('T Stage', ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
    n = st.selectbox('N Stage', ['N0', 'N1a', 'N1b'])
    m = st.selectbox('M Stage', ['M0', 'M1'])
    stage = st.selectbox('Overall Stage', ['I', 'II', 'III', 'IVA', 'IVB'])
    response = st.selectbox(
        'Treatment Response',
        ['Biochemical Incomplete', 'Excellent', 'Indeterminate', 'Structural Incomplete']
    )

    features = {
        'Age': age,
        'Gender': 1 if gender == 'Female' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'Hx Smoking': 1 if hx_smoking == 'Yes' else 0,
        'Hx Radiotherapy': 1 if hx_radiotherapy == 'Yes' else 0,
    }

    # One-hot encode categorical variables
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

    return pd.DataFrame([features])

# -------------------------------
# Prediction Logic
# -------------------------------
input_df = user_input_features()

# Ensure all columns exist
for col in x_train_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Match column order
input_df = input_df[x_train_columns]

if st.button('🔍 Predict'):
    probability = model.predict_proba(input_df)[:, 1][0]
    st.success(f"Predicted probability of recurrence: **{probability:.2f}**")

    if probability > 0.7:
        st.warning("⚠️ High risk of recurrence. Recommend further clinical evaluation.")
    elif probability > 0.4:
        st.info("🟠 Moderate risk. Consider regular follow-up and monitoring.")
    else:
        st.success("🟢 Low risk of recurrence.")
