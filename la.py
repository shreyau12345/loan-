import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("Loan Approval Prediction App")

# Load Dataset
df = pd.read_csv("loan_approval.csv")

# Features and Target
X = df[["Age", "Income", "CreditScore"]]
y = df["LoanApproved"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# User Input Section
st.header("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Income", min_value=1000, value=30000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)

# Prediction Button
if st.button("Predict Loan Approval"):
    input_data = pd.DataFrame([[age, income, credit_score]], 
                              columns=["Age", "Income", "CreditScore"])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")

        import streamlit as st

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://static.vecteezy.com/system/resources/previews/024/269/311/non_2x/car-house-personal-money-loan-concept-finance-business-icon-on-wooden-cube-saving-money-for-a-car-money-and-house-wooden-cubes-with-word-loan-copy-space-for-text-loan-payment-car-and-house-photo.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()