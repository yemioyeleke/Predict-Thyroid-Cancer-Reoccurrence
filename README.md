🩺 Project Overview

Thyroid cancer recurrence is a major clinical concern that affects post-treatment management and patient survival. This project applies data science and machine learning to analyze clinical and pathological data to predict the probability of recurrence in thyroid cancer patients.

Some ML models were trained; however, XGBoost achieved superior performance.

🎯 Objectives

To develop a predictive model capable of estimating thyroid cancer recurrence risk.

To compare the performance of Random Forest and XGBoost classifiers.

To visualize key metrics such as confusion matrix, ROC curve, and feature importance.

To deploy the final model as an interactive web application using Streamlit.

🧩 Key Features

Data preprocessing and cleaning

Handling of categorical variables via custom one-hot encoding

GridSearchCV for hyperparameter tuning

Cross-validation for robust model performance evaluation

Model persistence with pickle (.pkl)

Streamlit-based web interface for easy clinical interpretation

Probability-based predictions for clinical decision support

⚙️ Model Development Workflow

Data Preparation

Split into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Feature Engineering

Custom one-hot encoding for categorical columns like:

Thyroid Function

Physical Examination

Adenopathy

Pathology

Risk, Stage, T/N/M classification

Numerical features such as Age, Tumor Size, and Lymph Nodes are retained.

Model Training

Hyperparameter tuning using:

GridSearchCV(RF_model, param_grid, cv=5, scoring='accuracy')


Best parameters achieved:

{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
Best Score: 0.97


Evaluation Metrics

Confusion Matrix (visualized with seaborn heatmap)

Accuracy Score

ROC-AUC Curve

Cross-validation mean accuracy

Model Serialization

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

🌐 Web App (Streamlit)

The app provides a user-friendly interface for clinicians to input patient data and get recurrence probability predictions.

App Features:

Interactive input fields for clinical parameters

Real-time prediction output

Probability-based risk assessment (Low vs High Risk)

Deployed locally via Streamlit

Run the App:

cd app
streamlit run app.py


Example Output:

🩺 Predicted Probability of Recurrence: 0.72
⚠️ High Risk of Recurrence

📦 Installation Guide
1. Clone Repository
git clone https://github.com/<yourusername>/thyroid-recurrence-prediction.git
cd thyroid-recurrence-prediction/app

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # For Mac/Linux
venv\Scripts\activate          # For Windows

3. Install Dependencies
pip install -r requirements.txt

🧪 Requirements

Example requirements.txt:

streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
pickle-mixin

📊 Results Summary
Metric	Random Forest	XGBoost
Accuracy	0.97	0.98
AUC Score	0.95	0.97
Cross-Val Mean	0.96	0.97

XGBoost showed higher accuracy and generalization capacity.

🧬 Clinical Relevance

This project supports data-driven oncology by providing early warning of recurrence risks.
It does not replace medical judgment but serves as an assistive tool for:

Treatment planning

Patient follow-up prioritization

Research and educational purposes


🛡️ Disclaimer

This model is for research and educational purposes only.
Predictions should not be used as a standalone diagnostic tool.

📜 License

This project is distributed under the MIT License — feel free to use and modify with credit.
