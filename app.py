import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Predict Vehicle Losses",
    layout="centered"
)

# ---------------------------------
# Load model artifact
# ---------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("vehicle_loss_model.pkl")

artifact = load_artifact()
model = artifact["model"]
feature_names = artifact["feature_names"]

# ---------------------------------
# UI
# ---------------------------------
st.title("🚗🌧️ Predict Vehicle Losses (AI + Earth Observation)")
st.caption("Enter climate + vehicle info to predict Expected_Loss_USD.")
st.divider()

st.subheader("Inputs")

precipitation = st.number_input("Precipitation (mm)", 0.0, 500.0, 53.8)
temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 44.0)
base_loss_usd = st.number_input("Base_Loss_USD", 0, 1_000_000, 55_000)

flood_risk_flag = st.selectbox(
    "Flood Risk Flag",
    [0, 1],
    format_func=lambda x: "Low risk" if x == 0 else "Flood-prone"
)

prob_cover_deploy = st.slider("Prob_Cover_Deploy", 0.0, 1.0, 0.25)

# 🔴 THIS WAS MISSING
type_of_car = st.selectbox(
    "Type of Car",
    ["Sedan", "SUV", "Truck", "Bus", "Van", "Pickup"]
)

st.divider()

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔮 Predict Expected Loss", use_container_width=True):

    # Build raw input
    input_data = {
        "precipitation": precipitation,
        "temperature": temperature,
        "base_loss_usd": base_loss_usd,
        "flood_risk_flag": flood_risk_flag,
        "prob_cover_deploy": prob_cover_deploy,
        "type_of_car": type_of_car
    }

    X = pd.DataFrame([input_data])

    # Apply SAME encoding as training
    X = pd.get_dummies(X, columns=["type_of_car"], drop_first=True)

    # Add missing columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Ensure correct order
    X = X[feature_names]

    prediction = model.predict(X)[0]

    st.success(f"💰 **Expected Loss:** ${prediction:,.2f}")
