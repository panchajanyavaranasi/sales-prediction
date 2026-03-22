import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load Model Artifacts
# -----------------------------
# Streamlit Cloud working directory is the repo root
def load_model_from_github(url):
    response = requests.get(url)
    return pickle.loads(response.content)

BASE_URL = "https://github.com/panchajanyavaranasi/sales-prediction/tree/main/Sales_prediction/models"

try:
    model = load_model_from_github(BASE_URL + "knn_model.pkl")
    scaler = load_model_from_github(BASE_URL + "scaler.pkl")
    imputer = load_model_from_github(BASE_URL + "imputer.pkl")
    columns = load_model_from_github(BASE_URL + "columns.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load model from GitHub: {e}")
    st.stop()

st.title("📊 Sales Revenue Prediction")

st.write("Enter campaign details below:")

# -----------------------------
# Categorical Inputs
# -----------------------------
region = st.selectbox("Region", ["North", "South", "East", "West"])
channel = st.selectbox("Channel", ["Social Media", "Email", "TV", "SEO"])
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food"])
customer_segment = st.selectbox("Customer Segment", ["Retail", "Corporate", "Home Office"])

# -----------------------------
# Numerical Inputs
# -----------------------------
ad_spend = st.number_input("Ad Spend", min_value=0.0)
price = st.number_input("Price", min_value=0.0)
discount_rate = st.number_input("Discount Rate", min_value=0.0)
market_reach = st.number_input("Market Reach", min_value=0.0)
impressions = st.number_input("Impressions", min_value=0)
ctr = st.number_input("Click Through Rate", min_value=0.0)
competition_index = st.number_input("Competition Index", min_value=0.0)
seasonality_index = st.number_input("Seasonality Index", min_value=0.0)
campaign_days = st.number_input("Campaign Duration Days", min_value=0.0)
clv = st.number_input("Customer Lifetime Value", min_value=0.0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Revenue"):

    input_dict = {
        "region": region,
        "channel": channel,
        "product_category": product_category,
        "customer_segment": customer_segment,
        "ad_spend": ad_spend,
        "price": price,
        "discount_rate": discount_rate,
        "market_reach": market_reach,
        "impressions": impressions,
        "click_through_rate": ctr,
        "competition_index": competition_index,
        "seasonality_index": seasonality_index,
        "campaign_duration_days": campaign_days,
        "customer_lifetime_value": clv
    }

    input_df = pd.DataFrame([input_dict])

    # Separate types
    num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = input_df.select_dtypes(include=['object']).columns

    # Impute numeric
    input_df[num_cols] = imputer.transform(input_df[num_cols])

    # Encode categorical
    input_encoded = pd.get_dummies(input_df)

    # Align columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"💰 Predicted Revenue: ₹ {prediction[0]:,.2f}")