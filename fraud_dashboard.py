import streamlit as st
import pandas as pd
import pickle

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("transactions.csv")
df.columns = df.columns.str.strip()   # remove accidental spaces

# ============================
# LOAD MODEL
# ============================
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================
# LOAD TRAINING COLUMNS
# ============================
with open("training_columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

# ============================
# STREAMLIT UI
# ============================
st.title("Fraud Detection Dashboard")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Make a Prediction")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0)
transaction_type = st.selectbox("Transaction Type", df["TransactionType"].unique())
merchant = st.selectbox("Merchant Category", df["MerchantCategory"].unique())
location = st.selectbox("Location", df["Location"].unique())
device = st.selectbox("Device Type", df["DeviceType"].unique())

# Build input row
input_data = pd.DataFrame([{
    "TransactionAmount": amount,
    "TransactionType": transaction_type,
    "MerchantCategory": merchant,
    "Location": location,
    "DeviceType": device
}])

# ============================
# ENCODING
# ============================
input_encoded = pd.get_dummies(input_data)

# Align with training columns (CRITICAL FIX)
input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

# ============================
# PREDICT
# ============================
if st.button("Predict Fraud"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.error("⚠️ This transaction is predicted to be FRAUD.")
    else:
        st.success("✔️ This transaction is predicted to be LEGIT.")
