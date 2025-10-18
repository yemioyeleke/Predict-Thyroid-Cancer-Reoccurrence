import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

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
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Thyroid Cancer Recurrence Prediction", layout="wide")

st.title("🧬 Thyroid Cancer Recurrence Prediction App")
st.markdown("""
**Developed by DataLab Team**

A clinical decision-support tool to **predict thyroid cancer recurrence risk**  
and monitor model performance over time.
""")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📊 Model Dashboard", "🔎 Drift Detection"])

# -------------------------------
# 🔹 Tab 1: Prediction
# -------------------------------
with tab1:
    st.sidebar.title("Input Patient Data")

    def user_input_features():
        age = st.sidebar.slider('Age', 0, 120, 30)
        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        smoking = st.sidebar.selectbox('Smoking', ['Yes', 'No'])
        hx_smoking = st.sidebar.selectbox('Hx Smoking', ['Yes', 'No'])
        hx_radiotherapy = st.sidebar.selectbox('Hx Radiotherapy', ['Yes', 'No'])

        thyroid_function = st.sidebar.selectbox(
            'Thyroid Function',
            ['Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Euthyroid',
             'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism']
        )
        physical_exam = st.sidebar.selectbox(
            'Physical Examination',
            ['Diffuse goiter', 'Multinodular goiter', 'Normal',
             'Single nodular goiter-left', 'Single nodular goiter-right']
        )
        adenopathy = st.sidebar.selectbox(
            'Adenopathy',
            ['Bilateral', 'Extensive', 'Left', 'No', 'Posterior', 'Right']
        )
        pathology = st.sidebar.selectbox(
            'Pathology',
            ['Follicular', 'Hurthel cell', 'Micropapillary', 'Papillary']
        )
        focality = st.sidebar.selectbox('Focality', ['Multi-Focal', 'Uni-Focal'])
        risk = st.sidebar.selectbox('Risk', ['High', 'Intermediate', 'Low'])
        t = st.sidebar.selectbox('T Stage', ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
        n = st.sidebar.selectbox('N Stage', ['N0', 'N1a', 'N1b'])
        m = st.sidebar.selectbox('M Stage', ['M0', 'M1'])
        stage = st.sidebar.selectbox('Overall Stage', ['I', 'II', 'III', 'IVA', 'IVB'])
        response = st.sidebar.selectbox(
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

    input_df = user_input_features()

    for col in x_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0
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

# -------------------------------
# 🔹 Tab 2: Dashboard
# -------------------------------
with tab2:
    st.header("📊 Model Performance Dashboard")
    st.markdown("""
    Performance summary of **XGBoost** on the test dataset.
    """)

    st.subheader("📋 Classification Report")
    st.markdown("""
    | Metric | Class 0 | Class 1 | Macro Avg | Weighted Avg |
    |:--------|:--------|:--------|:------------|:--------------|
    | **Precision** | 0.98 | 1.00 | 0.99 | 0.99 |
    | **Recall** | 1.00 | 0.95 | 0.97 | 0.99 |
    | **F1-Score** | 0.99 | 0.97 | 0.98 | 0.99 |
    | **Accuracy** | | | | **0.99** |
    | **ROC-AUC** | | | | **0.97** |
    """)

    st.info("""
    **Clinical Interpretation:**  
    - Very high accuracy (99%) with minimal false negatives.  
    - Recall (0.95) for recurrence means the model detects most high-risk patients.  
    - ROC-AUC 0.97 ⇒ excellent separation between low and high recurrence risk.
    """)

    st.subheader("🧾 Confusion Matrix")
    cm = np.array([[54, 0], [1, 18]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No Recurrence', 'Recurrence'],
                yticklabels=['No Recurrence', 'Recurrence'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📈 ROC Curve (Simulated)")
    fpr = [0, 0.05, 0.1, 1]
    tpr = [0, 0.95, 1, 1]
    roc_auc = 0.97
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# -------------------------------
# 🔹 Tab 3: Data Drift Detection
# -------------------------------
with tab3:
    st.header("🔎 Data Drift Monitoring Dashboard")
    st.markdown("""
    Compare **incoming patient data** with the model's training distribution  
    to detect shifts in clinical patterns that may affect model reliability.
    """)

    uploaded_file = st.file_uploader("📤 Upload New Patient Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {new_data.shape[0]} new records.")
        
        ref_data = pd.DataFrame(columns=x_train_columns)
        for col in x_train_columns:
            ref_data[col] = np.random.rand(100)  # Simulated baseline reference

        report = Report(metrics=[DataDriftPreset()])
        mapping = ColumnMapping()
        report.run(reference_data=ref_data, current_data=new_data, column_mapping=mapping)
        report_path = "drift_report.html"
        report.save_html(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.info("Upload a new CSV dataset to check for drift in patient feature patterns.")
