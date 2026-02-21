import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load saved model and scaler
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud.")

# Input fields
transaction_id = st.text_input("Transaction ID", "")
transaction_date = st.text_input("Transaction Date (YYYY-MM-DD)", "")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
merchant_id = st.text_input("Merchant ID", "")
transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "Other"])
location = st.text_input("Location", "")

# ------------------------------
# Helper functions
# ------------------------------
def encode_text(val):
    """Manual encoding of categorical/text features to match training."""
    return hash(val) % 10000

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict Fraud"):

    # Create DataFrame in same column order as training
    df = pd.DataFrame({
        "TransactionID": [encode_text(transaction_id)],
        "TransactionDate": [encode_text(transaction_date)],
        "Amount": [amount],
        "MerchantID": [encode_text(merchant_id)],
        "TransactionType": [encode_text(transaction_type)],
        "Location": [encode_text(location)]
    })

    # Scale numeric column(s)
    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns)
    
    # Predict
    pred = model.predict(df.values)[0]
    prob = model.predict_proba(df.values)[0][1]  # probability of fraud

    # Display result
    st.markdown("### Result")
    if pred == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
    else:
        st.success(f"✅ Transaction is Safe. Fraud probability: {prob:.4f}")
