import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------
# LOAD DATA
# ---------------------------------
df = pd.read_csv("global_house.csv")

# Drop non-ML id columns
df.drop(columns=["property_id", "country", "city"], inplace=True)

# ---------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------
le_property = LabelEncoder()
le_furnish = LabelEncoder()

df["property_type"] = le_property.fit_transform(df["property_type"])
df["furnishing_status"] = le_furnish.fit_transform(df["furnishing_status"])

# ---------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------
X = df.drop(columns=["decision"])
y = df["decision"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# SCALE FEATURES
# ---------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ---------------------------------
# STREAMLIT UI
# ---------------------------------
st.title("🏡 Home Purchase Decision Predictor")

st.write("Enter property and customer details below:")

# ------- categorical inputs ------
property_type = st.selectbox("Property Type", le_property.classes_)
furnishing_status = st.selectbox("Furnishing Status", le_furnish.classes_)

# ------- numeric inputs ----------
property_size_sqft = st.number_input("Property Size (sqft)")
price = st.number_input("Property Price")
constructed_year = st.number_input("Constructed Year", min_value=1900, max_value=2025)
previous_owners = st.number_input("Previous Owners", step=1)
rooms = st.number_input("Rooms", step=1)
bathrooms = st.number_input("Bathrooms", step=1)
garage = st.number_input("Garage (0/1)", step=1)
garden = st.number_input("Garden (0/1)", step=1)
crime_cases_reported = st.number_input("Crime Cases", step=1)
legal_cases_on_property = st.number_input("Legal Issues (0/1)", step=1)
customer_salary = st.number_input("Monthly Salary")
loan_amount = st.number_input("Loan Amount")
loan_tenure_years = st.number_input("Loan Tenure (Years)")
monthly_expenses = st.number_input("Monthly Expenses")
down_payment = st.number_input("Down Payment")
emi_ratio = st.number_input("EMI to Income Ratio")
satisfaction_score = st.number_input("Satisfaction Score")
neighbourhood_rating = st.number_input("Neighbourhood Rating")
connectivity_score = st.number_input("Connectivity Score")

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------
if st.button("Predict Decision"):

    property_enc = le_property.transform([property_type])[0]
    furnish_enc = le_furnish.transform([furnishing_status])[0]

    input_data = np.array([[
        property_enc,
        furnish_enc,
        property_size_sqft,
        price,
        constructed_year,
        previous_owners,
        rooms,
        bathrooms,
        garage,
        garden,
        crime_cases_reported,
        legal_cases_on_property,
        customer_salary,
        loan_amount,
        loan_tenure_years,
        monthly_expenses,
        down_payment,
        emi_ratio,
        satisfaction_score,
        neighbourhood_rating,
        connectivity_score
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    result = "✅ THE CUSTOMER WILL BUY" if prediction == 1 else "❌ THE CUSTOMER NOT BUY"
    st.success(f"Prediction: {result}")
