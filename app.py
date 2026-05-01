import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Asteroid Threat Detection",
    page_icon="☄️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    return model, scaler, imputer

model, scaler, imputer = load_artifacts()

st.title("☄️ Asteroid Threat Detection")
st.write("Enter asteroid data below to find out if it is a potential threat.")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    dist_lunar = st.slider(
        "📏 Distance (Lunar Units)",
        0.0, 40.0, 15.0,
        help="1 lunar distance = 384,400 km. Dangerous threshold: 19.5 LD"
    )
with col2:
    velocity_km_s = st.slider(
        "💨 Velocity (km/s)",
        0.0, 40.0, 10.0,
        help="Speed of the asteroid relative to Earth at closest approach"
    )
with col3:
    absolute_magnitude = st.slider(
        "🔭 Absolute Magnitude",
        13.0, 35.0, 25.0,
        help="Lower number = larger asteroid. Dangerous threshold: 22.0 or below"
    )

st.markdown("""
### How it works
NASA classifies an asteroid as potentially hazardous if:
- It passes within **19.5 lunar distances** of Earth
- Its absolute magnitude is **22.0 or lower** (meaning it is large enough to cause damage)

Adjust the sliders above and click **Check Threat** to classify an asteroid.
""")

st.divider()

if st.button("Check Threat", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[dist_lunar, velocity_km_s, absolute_magnitude]],
        columns=['dist_lunar', 'velocity_km_s', 'absolute_magnitude']
    )
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ IMPACT: This asteroid is potentially hazardous.")
    else:
        st.success("✅ HALO: This asteroid is not a threat.")

st.divider()
st.markdown("### Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Impact Recall", "100%")
col2.metric("Impact Precision", "35%")
col3.metric("Overall Accuracy", "96%")
st.caption("Recall is prioritized: missing a real threat is worse than a false alarm.")