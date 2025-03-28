import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.title("Water Potability Prediction App")

# Create the "models" directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load models
@st.cache_resource
def load_models():
    try:
        if not os.path.exists('models/scaler.save'):
            st.error("scaler.save file not found in 'models' folder!")
            return None, None
        if not os.path.exists('models/final_rf_model.save'):
            st.error("random_forest_model.save file not found in 'models' folder!")
            return None, None
        scaler = joblib.load('models/scaler.save')
        final_rf_model = joblib.load('models/final_rf_model.save')
        return scaler, final_rf_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

scaler, final_rf_model = load_models()
if scaler is None or final_rf_model is None:
    st.stop()

# Safe ranges dictionary (WHO/EPA standards)
SAFE_RANGES = {
    'ph': (6.5, 8.5),
    'Hardness': (0, 150),
    'Solids': (0, 500),
    'Chloramines': (0, 4),
    'Sulfate': (0, 250),
    'Conductivity': (0, 800),
    'Organic_carbon': (0, 2),
    'Trihalomethanes': (0, 0.08),
    'Turbidity': (0, 1)
}

# Sidebar with safety indicators
st.sidebar.header("Water Quality Parameters")

def get_safety_status(value, param):
    min_val, max_val = SAFE_RANGES[param]
    if min_val <= value <= max_val:
        return "✅ Safe", "green"
    else:
        return "❌ Unsafe", "red"

# User Input Sliders
ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
status_text, status_color = get_safety_status(ph, 'ph')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (WHO: 6.5–8.5)</span>", unsafe_allow_html=True)

hardness = st.sidebar.slider("Hardness (mg/L)", 50.0, 300.0, 150.0)
status_text, status_color = get_safety_status(hardness, 'Hardness')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤150 mg/L)</span>", unsafe_allow_html=True)

solids = st.sidebar.slider("Total Dissolved Solids (mg/L)", 0.0, 1000.0, 300.0, step=10.0)
status_text, status_color = get_safety_status(solids, 'Solids')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤500 mg/L)</span>", unsafe_allow_html=True)

chloramines = st.sidebar.slider("Chloramines (mg/L)", 0.0, 8.0, 2.0, step=0.1)
status_text, status_color = get_safety_status(chloramines, 'Chloramines')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤4 mg/L)</span>", unsafe_allow_html=True)

sulfate = st.sidebar.slider("Sulfate (mg/L)", 100.0, 500.0, 300.0)
status_text, status_color = get_safety_status(sulfate, 'Sulfate')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤250 mg/L)</span>", unsafe_allow_html=True)

conductivity = st.sidebar.slider("Conductivity (µS/cm)", 100.0, 1000.0, 500.0)
status_text, status_color = get_safety_status(conductivity, 'Conductivity')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤800 µS/cm)</span>", unsafe_allow_html=True)

organic_carbon = st.sidebar.slider("Organic Carbon (mg/L)", 0.0, 5.0, 1.0, step=0.1)
status_text, status_color = get_safety_status(organic_carbon, 'Organic_carbon')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤2 mg/L)</span>", unsafe_allow_html=True)

trihalomethanes = st.sidebar.slider("Trihalomethanes (mg/L)", 0.0, 0.15, 0.05, step=0.01)
status_text, status_color = get_safety_status(trihalomethanes, 'Trihalomethanes')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤0.08 mg/L)</span>", unsafe_allow_html=True)

turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 2.0, 0.5, step=0.1)
status_text, status_color = get_safety_status(turbidity, 'Turbidity')
st.sidebar.markdown(f"<span style='color:{status_color}'>{status_text} (≤1 NTU)</span>", unsafe_allow_html=True)

# Create input data DataFrame
input_data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

# Check safety ranges
unsafe_params = [param for param, value in input_data.iloc[0].items() if not (SAFE_RANGES[param][0] <= value <= SAFE_RANGES[param][1])]

# Make prediction if safe
if unsafe_params:
    st.error("## ❌ Water is Not Potable!")
    st.warning(f"**Unsafe parameters detected:** {', '.join(unsafe_params)}")
    final_prediction = 0
else:
    scaled_input = scaler.transform(input_data)
    prediction = final_rf_model.predict(scaled_input)
    final_prediction = prediction[0]

# Display final prediction
st.subheader("Final Prediction")
if not unsafe_params or final_prediction == 1:
    st.success("## ✅ The water is Potable!")
else:
    st.error("## ❌ The water is Not Potable")