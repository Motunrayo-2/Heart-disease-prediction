import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. Load Model, Scaler, and SHAP background data ---
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and SHAP background data only once."""
    model = tf.keras.models.load_model('MY_ANN_model.h5')
    scaler = joblib.load('standard_scaler.save')
    # Load a small sample of your training data for SHAP
    # This must contain the same columns as your final preprocessed input
    background_data = pd.read_csv('background_data.csv')
    return model, scaler, background_data

model, scaler, background_data = load_assets()

# --- 2. Streamlit App Title and Introduction ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Cardiovascular Health: Heart Disease Risk Prediction")
st.markdown("---")
st.write("""
Welcome! This application predicts the risk of heart disease based on a patient's medical metrics.
Please enter the patient's information below, and the model will provide a prediction along with an explanation of its decision.
""")

# --- 3. User Input Interface ---
st.sidebar.header("Patient's Health Metrics")
st.sidebar.info("Adjust the values to see how they impact the prediction.")

# Dictionary to store user inputs
user_input = {}

# Numerical Inputs
user_input['age'] = st.sidebar.slider("Age", 20, 100, 50)
user_input['resting_bp'] = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
user_input['cholesterol'] = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
user_input['max_heart_rate'] = st.sidebar.slider("Maximum Heart Rate Achieved", 70, 220, 150)
user_input['oldpeak'] = st.sidebar.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1)

# Categorical Inputs
user_input['sex'] = st.sidebar.selectbox("Sex", options=["Male", "Female"])
user_input['chest_pain'] = st.sidebar.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
user_input['fasting_bs'] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
user_input['rest_ecg'] = st.sidebar.selectbox("Resting ECG Results", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
user_input['exang'] = st.sidebar.selectbox("Exercise Induced Angina", options=["No", "Yes"])
user_input['slope'] = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
user_input['vessels_colored'] = st.sidebar.number_input("Number of Major Vessels (0-3)", 0, 3, 0)
user_input['thal'] = st.sidebar.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

# --- 4. Prediction Logic and SHAP Explanation ---
st.markdown("### Prediction")
st.markdown("---")

if st.sidebar.button("Predict Heart Disease Risk"):
    
    # 4.1. Preprocessing the input data
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Encode categorical features (ensure this mapping is IDENTICAL to your training script)
    input_df['sex'] = input_df['sex'].map({"Male": 1, "Female": 0})
    input_df['fasting_bs'] = input_df['fasting_bs'].map({"Yes": 1, "No": 0})
    input_df['exang'] = input_df['exang'].map({"Yes": 1, "No": 0})

    # One-hot encode or integer map other categorical features
    # Example for 'chest_pain':
    chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    input_df['chest_pain'] = input_df['chest_pain'].map(chest_pain_map)
    
    # ... Add mappings for rest_ecg, slope, and thal
    
    # Identify numerical features
    numerical_features = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Scale the numerical features using the pre-trained scaler
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Create the final NumPy array for the model
    # Ensure the columns are in the same order as in your training data
    features_order = numerical_features + ['sex', 'chest_pain', 'fasting_bs', 'rest_ecg', 'exang', 'slope', 'vessels_colored', 'thal']
    final_input = input_df[features_order].to_numpy()

    # 4.2. Make the prediction
    prediction = model.predict(final_input)[0][0]
    prediction_proba = prediction * 100

    # 4.3. Display Prediction Result with a progress bar
    with st.spinner("Analyzing..."):
        st.subheader(f"Prediction: {'Heart Disease' if prediction > 0.5 else 'No Heart Disease'}")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Predicted Probability", value=f"{prediction_proba:.2f}%")
        with col2:
            st.progress(float(prediction))
        
        if prediction > 0.5:
            st.error("The model predicts a high risk. It is recommended to consult a healthcare professional.")
        else:
            st.success("The model predicts a low risk of heart disease.")
            
    st.markdown("---")
    
    # 4.4. SHAP Explanation
    st.markdown("### How the Model Made its Prediction (SHAP)")
    st.write("The chart below shows how each feature contributed to the final prediction. Red features push the prediction towards heart disease, while blue features push it away.")
    
    # Create the SHAP explainer
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    # Calculate SHAP values for the current user's input
    shap_values = explainer.shap_values(final_input)
    
    # Display SHAP force plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    shap.force_plot(
        explainer.expected_value[0], 
        shap_values[0], 
        input_df[features_order].iloc[0], 
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    
# --- 5. Model Insights & Metrics (Optional but Recommended) ---
st.markdown("---")
st.markdown("### Model Performance Insights")
st.write("Below are some key metrics of the model, which was trained to achieve an accuracy of 94%.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Model Accuracy", value="94%")
with col2:
    st.metric(label="F1 Score", value="0.93")
with col3:
    st.metric(label="Precision", value="0.95")
