import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration and Styling
st.set_page_config(
    page_title="Lung Cancer Risk Screening Engine",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Lung Cancer Risk Screening Engine")
st.markdown("""
This web interface utilizes a fine-tuned **Random Forest Classifier** to assess predictive lung cancer risk based on a patient's behavioral habits, demographics, and clinical symptoms. 
*Please fill out the screening questionnaire below to generate a diagnostic probability score.*
""")
st.write("---")

# 2. Cached Function to Load the Model (Ensures fast performance)
@st.cache_resource
def load_ml_model():
    # Loads the serialized file we saved earlier
    return joblib.load('best_lung_cancer_rf_model.pkl')

try:
    model = load_ml_model()
except Exception as e:
    st.error("⚠️ Error loading the machine learning model file. Ensure 'best_lung_cancer_rf_model.pkl' is in the same directory.")
    st.stop()

# 3. Create Form Layout for Survey Inputs
st.subheader("📋 Patient Screening Questionnaire")

with st.form("screening_form"):
    # Group inputs into columns for clean structure
    col1, col2 = st.columns(2)
    
    with col1:
        gender_input = st.selectbox("Patient Gender", options=["Male", "Female"])
        age = st.slider("Patient Age", min_value=1, max_value=100, value=50)
        smoking = st.selectbox("Smoking Habit", options=["Yes", "No"])
        yellow_fingers = st.selectbox("Yellow Fingers symptom", options=["Yes", "No"])
        anxiety = st.selectbox("Experiences Anxiety", options=["Yes", "No"])
        peer_pressure = st.selectbox("Subject to Peer Pressure", options=["Yes", "No"])
        chronic_disease = st.selectbox("History of Chronic Disease", options=["Yes", "No"])
    
    with col2:
        fatigue = st.selectbox("Experiences Fatigue", options=["Yes", "No"])
        allergy = st.selectbox("History of Allergies", options=["Yes", "No"])
        wheezing = st.selectbox("Experiences Wheezing", options=["Yes", "No"])
        alcohol = st.selectbox("Consumes Alcohol", options=["Yes", "No"])
        coughing = st.selectbox("Frequent/Severe Coughing", options=["Yes", "No"])
        shortness_breath = st.selectbox("Shortness of Breath", options=["Yes", "No"])
        swallowing_diff = st.selectbox("Difficulty Swallowing", options=["Yes", "No"])
        chest_pain = st.selectbox("Frequent Chest Pain", options=["Yes", "No"])

    # Submit button
    submit_button = st.form_submit_button(label="Generate Risk Assessment")

# 4. Handle Form Submission and Make Prediction
if submit_button:
    # Map user interface responses to the exact numerical encoding used during model training
    # Standard survey encoding mapping: Yes = 2, No = 1. Male = 1, Female = 0.
    binary_map = {"Yes": 2, "No": 1}
    gender_map = {"Male": 1, "Female": 0}
    
    # Constructing the exact data dictionary 
    # NOTE: Check if your training columns have trailing spaces like 'FATIGUE ' or 'ALLERGY '
    # Constructing the exact data dictionary 
    # Mapped perfectly to match the 'Feature names seen at fit time' from your error log
    patient_data = {
        'GENDER': gender_map[gender_input],
        'AGE': age,
        'SMOKING': binary_map[smoking],
        'YELLOW_FINGERS': binary_map[yellow_fingers],
        'ANXIETY': binary_map[anxiety],
        'PEER_PRESSURE': binary_map[peer_pressure],
        'CHRONIC DISEASE': binary_map[chronic_disease],       # Changed from underscore to Space
        'FATIGUE': binary_map[fatigue],                       # Removed trailing space
        'ALLERGY': binary_map[allergy],                       # Removed trailing space
        'WHEEZING': binary_map[wheezing],
        'ALCOHOL CONSUMING': binary_map[alcohol],             # Changed from underscore to Space
        'COUGHING': binary_map[coughing],
        'SHORTNESS OF BREATH': binary_map[shortness_breath],   # Changed from underscores to Spaces
        'SWALLOWING DIFFICULTY': binary_map[swallowing_diff], # Changed from underscore to Space
        'CHEST PAIN': binary_map[chest_pain]                  # Changed from underscore to Space
    }
    
    # Convert dictionary into a Single Row DataFrame for prediction pipeline
    input_df = pd.DataFrame([patient_data])
    
    # Execute Model Inference
    risk_probability = model.predict_proba(input_df)[0][1]
    risk_percentage = risk_probability * 100
    
    st.write("---")
    st.subheader("📊 Diagnostic Screening Output")
    
    # 5. Display Custom Colored Alerts Based on Risk Severity Thresholds
    if risk_percentage >= 50.0:
        st.error(f"🚨 **High Risk Detected**")
        st.metric(label="Lung Cancer Risk Probability", value=f"{risk_percentage:.2f}%")
        st.markdown("> **Clinical Note:** The predictive model indicates a high probability matching historical clinical positive cases. Immediate clinical validation and diagnostic imaging (e.g., Low-Dose CT Scan) are highly recommended.")
    else:
        st.success(f"✅ **Low Risk Detected**")
        st.metric(label="Lung Cancer Risk Probability", value=f"{risk_percentage:.2f}%")
        st.markdown("> **Clinical Note:** The model predicts a low match with symptomatic lung cancer profiles. Continue regular healthy lifestyle management and routine checkups.")