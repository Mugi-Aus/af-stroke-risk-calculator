# app.py - Run via command terminal: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="AF Stroke Risk Engine", layout="centered")

# --- 1. APPLICATION HEADER ---
st.title("🫀 Atrial Fibrillation Stroke Survival Predictor")
st.markdown("Enter patient metrics below to compute 1-year and 5-year stroke-free survival percentages across multiple ML ensembles.")

# --- 2. SIDEBAR PATIENT CONFIGURATION INPUTS ---
st.sidebar.header("📋 Patient Clinical Metrics")
input_age = st.sidebar.slider("Patient Baseline Age", min_value=40, max_value=90, value=72, step=1)
input_diabetes = st.sidebar.selectbox("Diabetes Mellitus History", options=["No", "Yes"], index=1)
input_smoking = st.sidebar.selectbox("Active Smoking History", options=["No", "Yes"], index=0)
input_hypertension = st.sidebar.selectbox("Systemic Hypertension History", options=["No", "Yes"], index=1)

# Map human-readable selections back to binary data matrix numbers (0 or 1)
patient_data = {
    'age': [input_age],
    'diabetes': [1 if input_diabetes == "Yes" else 0],
    'smoking': [1 if input_smoking == "Yes" else 0],
    'hypertension': [1 if input_hypertension == "Yes" else 0]
}
new_patient_df = pd.DataFrame(patient_data)

# --- 3. BACKGROUND MODEL LOADING UTILITY ---
@st.cache_resource # Keeps models in memory so they load only once
def load_trained_pipelines():
    with open('rsf_model.pkl', 'rb') as f:
        rsf = pickle.load(f)
    xgb_booster = xgb.Booster()
    xgb_booster.load_model('xgb_model.json')
    lgb_booster = lgb.Booster(model_file='lgb_model.txt')
    with open('baseline_cum_hazard.pkl', 'rb') as f:
        baseline_cum_hazard = pickle.load(f)
    return rsf, xgb_booster, lgb_booster, baseline_cum_hazard

# To run this script locally, ensure your saved model files are inside the same folder!
try:
    rsf, xgb_model, lgb_model, baseline_cum_hazard = load_trained_pipelines()
    
    # --- 4. EXECUTE CALCULATIONS ON SLIDER ADJUSTMENTS ---
    # RSF Execution
    rsf_f = rsf.predict_survival_function(new_patient_df)[0]
    rsf_1 = rsf_f(12) * 100
    rsf_5 = rsf_f(60) * 100

    # XGBoost Execution
    dpt = xgb.DMatrix(new_patient_df)
    pt_hr = np.exp(xgb_model.predict(dpt))[0]
    s0_12 = np.exp(-baseline_cum_hazard[max([t for t in baseline_cum_hazard.keys() if t <= 12])])
    s0_60 = np.exp(-baseline_cum_hazard[max([t for t in baseline_cum_hazard.keys() if t <= 60])])
    xgb_1 = np.power(s0_12, pt_hr) * 100
    xgb_5 = np.power(s0_60, pt_hr) * 100

    # LightGBM Execution
    lgb_risk = lgb_model.predict(new_patient_df)[0]
    lgb_5 = (1 - lgb_risk) * 100

    # --- 5. RENDER RE-CALCULATED RESULTS TO WEB DASHBOARD ---
    st.subheader("📊 Individual Prognosis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model 1: RSF (5-Year)", value=f"{rsf_5:.1f}%")
        st.caption(f"1-Year Survival: {rsf_1:.1f}%")
    with col2:
        st.metric(label="Model 2: XGBoost (5-Year)", value=f"{xgb_5:.1f}%")
        st.caption(f"1-Year Survival: {xgb_1:.1f}%")
    with col3:
        st.metric(label="Model 3: LightGBM (5-Year)", value=f"{lgb_5:.1f}%")
        st.caption("1-Year Survival: N/A")

except FileNotFoundError:
    st.error("⚠️ Pre-trained model files not found in root path directory. Please download your model weight files (.pkl, .json, .txt) from your Colab runtime session into this folder to see the app calculate.")
