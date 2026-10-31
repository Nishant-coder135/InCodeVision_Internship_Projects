import json
import os
import joblib
import pandas as pd
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & GLOBAL CONFIGURATION INGESTION
# =====================================================================
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_application_artifacts():
    """Loads the locked schema, fitted pipelines, and optimized models.
    
    Handles directory boundaries dynamically if deployed in cloud microservices.
    """
    config_path = "config/config.json"
    model_path = "artifacts/churn_model.pkl"
    preprocessor_path = "artifacts/preprocessor.joblib"
    
    # Path resolution for subfolder context isolation
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
        model_path = os.path.join("..", model_path)
        preprocessor_path = os.path.join("..", preprocessor_path)
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return config, model, preprocessor

try:
    CONFIG, MODEL, PREPROCESSOR = load_application_artifacts()
    THRESHOLD = CONFIG["inference_settings"]["default_decision_threshold"]
except Exception as e:
    st.error(f"Critical System Ingestion Failure: Missing pipeline dependencies. Error: {e}")
    st.stop()

# =====================================================================
# 2. APPLICATION INTERFACE HEADERS
# =====================================================================
st.title("📊 Customer Churn Risk Prediction Dashboard")
st.markdown("""
This production-grade interface computes proactive churn trajectories for subscriber accounts.
It leverages an optimized **Random Forest Classifier** tuned via a Precision-Recall curve boundary to balance retention budgets.
""")

st.divider()

# =====================================================================
# 3. INDIVIDUAL ACCOUNT INGESTION PANEL & SIDEBAR FORM
# =====================================================================
st.subheader("💡 Individual Account Ingestion Panel")
st.info("Adjust the customer metrics in the left sidebar configuration panel to view live risk scores.")

# Build form entries based dynamically on your 13 dataset traits
with st.sidebar:
    st.header("📋 Client Parameters Input")
    
    # Categorical Business Flags
    complains = st.selectbox("Customer Raised Complaints?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    tariff_plan = st.selectbox("Tariff Arrangement Plan:", options=[0, 1], format_func=lambda x: "Contract Plan (1)" if x == 1 else "Pay-as-you-go (0)")
    call_failure = st.slider("Experienced Call Failures (Count):", min_value=0, max_value=40, value=2)
    charge_amount = st.slider("Billing Value Tier (0 Lowest - 9 Highest):", min_value=0, max_value=9, value=5)
    age_group = st.slider("Age Group Cohort Bucket (1-5):", min_value=1, max_value=5, value=3)
    
    st.divider()
    
    # Continuous Numerical Usage Traits
    sub_length = st.number_input("Subscription Lifespan (Months):", min_value=1, max_value=100, value=12)
    seconds_use = st.number_input("Voice Seconds Accumulation (Duration):", min_value=0, max_value=20000, value=3500)
    sms_freq = st.number_input("Text Messaging Volume (SMS Count):", min_value=0, max_value=1000, value=150)
    called_nums = st.number_input("Distinct Dialed Contacts (Count):", min_value=0, max_value=200, value=25)
    age_val = st.number_input("Actual Customer Age (Years):", min_value=18, max_value=100, value=35)
    cust_value = st.number_input("Calculated Customer Lifetime Value Equity:", min_value=0.0, max_value=1000.0, value=250.0)

# Compile incoming sidebar input vector cleanly into an aligned DataFrame
input_payload = pd.DataFrame([{
    "Call  Failure": call_failure,
    "Complains": complains,
    "Subscription  Length": sub_length,
    "Charge  Amount": charge_amount,
    "Seconds of Use": seconds_use,
    "Frequency of SMS": sms_freq,
    "Distinct Called Numbers": called_nums,
    "Age Group": age_group,
    "Tariff Plan": tariff_plan,
    "Age": age_val,
    "Customer Value": cust_value
}])

# Execute dynamic pipeline inferences
try:
    # The data automatically maps through features.py and ColumnTransformers in one step
    processed_vector = PREPROCESSOR.transform(input_payload)
    churn_probability = MODEL.predict_proba(processed_vector)[0, 1]
    is_churn = 1 if churn_probability >= THRESHOLD else 0
    
    # Display performance score metrics using design components
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Exit Probability", value=f"{churn_probability * 100:.2f} %")
        
    with col2:
        if is_churn == 1:
            st.error(f"🚨 CRITICAL ALERT: High Churn Risk Flagged (Cutoff: {THRESHOLD:.2f})")
        else:
            st.success("✅ STABLE STATUS: Customer Classified as Low Retention Risk")
            
    # Primary factor visualization logic block
    st.subheader("🔍 Core Driver Risk Telemetry")
    if complains == 1:
        st.warning("⚠️ High Risk Vector: Customer has an unresolved active complaint log. Priority customer retention action required.")
    if call_failure > 15:
        st.warning(f"⚠️ High Risk Vector: Customer experienced extreme network drop friction ({call_failure} call failures). Check area infrastructure.")
        
except Exception as e:
    st.error(f"Inference Mapping Error: Data layout did not match your production pipeline model specs. Details: {e}")
