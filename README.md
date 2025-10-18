# 🩺 Thyroid Cancer Recurrence Prediction

## 🩸 Project Overview

Thyroid cancer recurrence is a major clinical concern that affects post-treatment management and patient survival.  
This project applies **data science and machine learning** to analyze clinical and pathological data to **predict the probability of recurrence** in thyroid cancer patients.

Several ML models were trained; however, **XGBoost achieved superior performance**.

---

## 🎯 Objectives

- Develop a predictive model capable of estimating thyroid cancer recurrence risk  
- Compare the performance of various **classifiers**   
- Visualize key metrics such as **confusion matrix**, **ROC curve**, and **feature importance**  
- Deploy the final model as an **interactive web application** using **Streamlit**

---

## 🧩 Key Features

- Data preprocessing and cleaning  
- Handling of categorical variables via custom one-hot encoding  
- GridSearchCV for hyperparameter tuning  
- Cross-validation for robust model performance evaluation  
- Model persistence with pickle (`.pkl`)  
- Streamlit-based web interface for easy clinical interpretation  
- Probability-based predictions for clinical decision support  

---

## ⚙️ Model Development Workflow

### 1. Data Preparation

Split into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## 🧩 2. Feature Engineering

Custom **one-hot encoding** was applied for categorical columns such as:

- Thyroid Function  
- Physical Examination  
- Adenopathy  
- Pathology  
- Risk, Stage, T/N/M classification  

Numerical features such as **Age**, **Tumor Size**, and **Lymph Nodes** were retained for model training.

---

## ⚙️ 3. Model Training

**Hyperparameter tuning** was performed using `GridSearchCV`:

```python
GridSearchCV(RF_model, param_grid, cv=5, scoring='accuracy')
Best parameters achieved:

python
Copy code
{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
Best Score: 0.97


## 📏 4. Evaluation Metrics

Model performance was assessed using:

- **Confusion Matrix** (visualized with seaborn heatmap)  
- **Accuracy Score**  
- **ROC-AUC Curve**  
- **Cross-validation Mean Accuracy**  

---

## 💾 5. Model Serialization

The trained **XGBoost model** was saved using `pickle` for later deployment:

```python
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)


🌐 Web App (Streamlit)

The app provides a user-friendly interface for clinicians to input patient data and receive recurrence probability predictions.

🧠 App Features

- Interactive input fields for clinical parameters

- Real-time prediction output

- Probability-based risk assessment (Low vs High Risk)

- Deployed locally using Streamlit


▶️ Run the App
cd app
streamlit run app.py
