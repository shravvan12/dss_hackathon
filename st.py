import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess data
data = pd.read_csv("HR_Analytics.csv")
data['YearsWithCurrManager'].fillna(data['YearsWithCurrManager'].median(), inplace=True)
data.drop(columns=['EmpID', 'EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'], inplace=True)

# Label encoding
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
reverse_encodings = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    reverse_encodings[col] = dict(zip(le.transform(le.classes_), le.classes_))

# Performance prediction model
X_p = data.drop(columns=['PerformanceRating', 'Attrition'])
y_p = data['PerformanceRating']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)

performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
performance_model.fit(X_train_p, y_train_p)
expected_perf_features = X_p.columns.tolist()

# Attrition prediction model
X_ret = data.drop(columns=['Attrition', 'PerformanceRating'])
y_ret = data['Attrition']
X_ret_train, X_ret_test, y_ret_train, y_ret_test = train_test_split(X_ret, y_ret, test_size=0.2, random_state=42)

attrition_model = RandomForestClassifier(n_estimators=100, random_state=42)
attrition_model.fit(X_ret_train, y_ret_train)
expected_attrition_features = X_ret.columns.tolist()

# Streamlit app
st.title("Employee Performance & Attrition Prediction")

st.header("Enter Essential Employee Info:")

# Essential fields only
age = st.slider("Age", 18, 60, 30)
job_role = st.selectbox("Job Role", label_encoders["JobRole"].classes_)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
overtime = st.selectbox("OverTime", label_encoders["OverTime"].classes_)
marital_status = st.selectbox("Marital Status", label_encoders["MaritalStatus"].classes_)
years_at_company = st.slider("Years at Company", 0, 40, 5)

# Build a full input row with defaults + user inputs
default_input = X_p.iloc[0].copy()

# Overwrite essentials
default_input["Age"] = age
default_input["MonthlyIncome"] = monthly_income
default_input["JobSatisfaction"] = job_satisfaction
default_input["YearsAtCompany"] = years_at_company
default_input["JobRole"] = label_encoders["JobRole"].transform([job_role])[0]
default_input["OverTime"] = label_encoders["OverTime"].transform([overtime])[0]
default_input["MaritalStatus"] = label_encoders["MaritalStatus"].transform([marital_status])[0]

input_df_perf = pd.DataFrame([default_input])
input_df_attr = pd.DataFrame([default_input])

input_df_perf = input_df_perf[expected_perf_features]
input_df_attr = input_df_attr[expected_attrition_features]

if st.button("Predict Performance", key="perf_btn"):
    perf_pred = performance_model.predict(input_df_perf)[0]
    st.success(f"Predicted Performance Rating: {round(perf_pred, 2)}")

if st.button("Predict Attrition", key="attr_btn"):
    attr_pred = attrition_model.predict(input_df_attr)[0]
    attr_label = reverse_encodings["Attrition"][attr_pred]
    st.success(f"Predicted Attrition: {attr_label}")
