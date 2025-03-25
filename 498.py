import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🚀 Step 1: Generate Dummy Data

np.random.seed(42)
n_customers = 10000  # Number of rural customers

# Creating a dataset for rural finance
rural_finance_df = pd.DataFrame({
    'customer_id': np.arange(1, n_customers+1),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
    'loan_amount': np.random.uniform(5000, 100000, n_customers),
    'loan_purpose': np.random.choice(['Agriculture', 'Small Business', 'Education', 'Personal'], n_customers),
    'credit_score': np.random.uniform(300, 850, n_customers),
    'monthly_income': np.random.uniform(5000, 50000, n_customers),
    'existing_loans': np.random.randint(0, 5, n_customers),
    'loan_tenure': np.random.randint(12, 60, n_customers),  # Loan tenure in months
    'default_risk': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),  # 1 means high risk of default
    'late_payment_flag': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])  # 1 means late payment history
})

# 🚀 Step 2: Customer Acquisition Model (Loan Approval Prediction)
X_acquisition = rural_finance_df[['loan_amount', 'credit_score', 'monthly_income', 'existing_loans', 'loan_tenure']]
y_acquisition = rural_finance_df['default_risk']

X_acq_train, X_acq_test, y_acq_train, y_acq_test = train_test_split(X_acquisition, y_acquisition, test_size=0.3, random_state=42)

loan_approval_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
loan_approval_model.fit(X_acq_train, y_acq_train)

y_acq_pred = loan_approval_model.predict(X_acq_test)
print("Customer Acquisition Model Accuracy:", accuracy_score(y_acq_test, y_acq_pred))

# 🚀 Step 3: Early Warning System (High-Risk Loan Prediction)
X_ews = rural_finance_df[['credit_score', 'existing_loans', 'loan_amount', 'monthly_income']]
y_ews = rural_finance_df['default_risk']

X_ews_train, X_ews_test, y_ews_train, y_ews_test = train_test_split(X_ews, y_ews, test_size=0.3, random_state=42)

ews_model = RandomForestClassifier(n_estimators=100, random_state=42)
ews_model.fit(X_ews_train, y_ews_train)

y_ews_pred = ews_model.predict(X_ews_test)
print("Early Warning System Accuracy:", accuracy_score(y_ews_test, y_ews_pred))

# 🚀 Step 4: Collection Strategy Optimization (Late Payment Prediction)
X_collection = rural_finance_df[['loan_amount', 'credit_score', 'monthly_income', 'loan_tenure']]
y_collection = rural_finance_df['late_payment_flag']

X_col_train, X_col_test, y_col_train, y_col_test = train_test_split(X_collection, y_collection, test_size=0.3, random_state=42)

collection_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
collection_model.fit(X_col_train, y_col_train)

y_col_pred = collection_model.predict(X_col_test)
print("Collection Strategy Model Accuracy:", accuracy_score(y_col_test, y_col_pred))

# 🚀 Step 5: Save Data and Models
rural_finance_df.to_csv("rural_finance_data.csv", index=False)

joblib.dump(loan_approval_model, "loan_approval_model.pkl")
joblib.dump(ews_model, "early_warning_system.pkl")
joblib.dump(collection_model, "collection_strategy_model.pkl")

print("Data and Models Saved Successfully!")
