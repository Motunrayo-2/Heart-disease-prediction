import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import matplotlib.pyplot as plt

# --- 1. Load Assets ---
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and SHAP background data only once."""
    try:
        model = tf.keras.models.load_model('MY_ANN_model.h5')
        scaler = joblib.load('scaler.joblib')
        background_data = pd.read_csv('background_data(1).csv')
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure all necessary files (model.h5, scaler.joblib, background_data.csv) are in the app's directory.")
        st.stop()
    return model, scaler, background_data

model, scaler, background_data = load_assets()

# --- 2. App Title and Introduction ---
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

# Create input fields for each feature, using the correct column names from your dataset
user_input = {
    'age': st.sidebar.slider("Age", 20, 100, 50),
    'sex': st.sidebar.selectbox("Sex", options=["Male", "Female"]),
    'cp': st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3]),
    'trestbps': st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120),
    'chol': st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200),
    'fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"]),
    'restecg': st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2]),
    'thalach': st.sidebar.slider("Maximum Heart Rate Achieved", 70, 220, 150),
    'exang': st.sidebar.selectbox("Exercise Induced Angina", options=["No", "Yes"]),
    'oldpeak': st.sidebar.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1),
    'slope': st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2]),
    'ca': st.sidebar.number_input("Number of Major Vessels (0-3)", 0, 3, 0),
    'thal': st.sidebar.selectbox("Thalassemia", options=[0, 1, 2]),
}

# Prediction Button
if st.sidebar.button("Predict Heart Disease Risk"):
    
    # --- 4. Preprocessing the User's Inputs ---
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Convert selectbox strings to numerical values
    input_df['sex'] = input_df['sex'].map({"Male": 1, "Female": 0})
    input_df['fbs'] = input_df['fbs'].map({"Yes": 1, "No": 0})
    input_df['exang'] = input_df['exang'].map({"Yes": 1, "No": 0})
    
    # One-hot encode the categorical features
    # This must be done BEFORE scaling, and must use the exact same column names
    # and logic as your Colab notebook.
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
    
    # Define the final column order
    # This order must be IDENTICAL to the columns in your background_data.csv
    final_columns = background_data.columns.tolist()
    
    # Align the input DataFrame to the correct column order and add missing columns with 0
    input_aligned = input_df.reindex(columns=final_columns, fill_value=0)
    
    # Scale the numerical features using the pre-trained scaler
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    input_aligned[numerical_features] = scaler.transform(input_aligned[numerical_features])
    
    # --- 5. Make Prediction and Display Results ---
    with st.spinner("Analyzing..."):
        # Convert to numpy array for model prediction
        final_input_array = input_aligned.to_numpy()
        prediction = model.predict(final_input_array)[0][0]
        prediction_proba = prediction * 100

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
    
    # --- 6. SHAP Explanation ---
    st.markdown("### How the Model Made its Prediction (SHAP)")
    st.write("The chart below shows how each feature contributed to the final prediction. Red features push the prediction towards heart disease, while blue features push it away.")
    
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(input_aligned)
    
    # Display SHAP force plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    shap.force_plot(
        explainer.expected_value[0], 
        shap_values[0], 
        input_aligned, # Pass the DataFrame here for correct labeling
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    
    # --- 7. Static Model Insights & Metrics ---
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
