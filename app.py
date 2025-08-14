import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import time

# --- 1. Load Assets ---
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and SHAP background data only once."""
    try:
        model = tf.keras.models.load_model('MY_ANN_model.h5')
        scaler = joblib.load('scaler.joblib')
        background_data = pd.read_csv('background_data.csv')
        df = pd.read_csv('heart.csv') # Load original data for insights page
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure all necessary files (model.h5, scaler.joblib, background_data.csv, heart_disease.csv) are in the app's directory.")
        st.stop()
    return model, scaler, background_data, df

model, scaler, background_data, df = load_assets()

# --- 2. Custom CSS for a Modern Look ---
def custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            width: 100%;
            border-radius: 25px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #4CAF50;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            border-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
            padding: 20px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
        }
        h1, h2, h3 {
            color: #333333;
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 3. Session State Management for Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'input_aligned' not in st.session_state:
    st.session_state.input_aligned = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None

# --- 4. Page Functions ---
def intro_page():
    """First page: Introduction to the app."""
    st.title("Cardiovascular Health: Heart Disease Risk Prediction")
    st.markdown("---")
    st.write("Welcome! This application predicts the risk of heart disease based on a patient's medical metrics.")
    st.write("To get started, please click the button below to input the patient's information.")
    
    # Place your image file (e.g., 'image.png') in the same directory as this script.
    # st.image('image.png', use_column_width=True) # User's requested code
    # Using a placeholder for demonstration. Change 'image.png' to your file name.
    st.image('https://images.unsplash.com/photo-1576091160550-2173f47c09d9?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D', use_column_width=True)


    st.markdown("---")
    if st.button("Start Inputting Features"):
        st.session_state.page = 'input_form'

def input_form_page():
    """Second page: User input form for features."""
    st.title("Patient's Health Metrics")
    st.markdown("---")
    
    # Create input fields for each feature
    user_input = {
        'age': st.slider("Age", 20, 100, 50),
        'sex': st.selectbox("Sex", options=["Male", "Female"]),
        'cp': st.selectbox("Chest Pain Type", options=[0, 1, 2, 3]),
        'trestbps': st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120),
        'chol': st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200),
        'fbs': st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"]),
        'restecg': st.selectbox("Resting ECG Results", options=[0, 1, 2]),
        'thalach': st.slider("Maximum Heart Rate Achieved", 70, 220, 150),
        'exang': st.selectbox("Exercise Induced Angina", options=["No", "Yes"]),
        'oldpeak': st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1),
        'slope': st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2]),
        'ca': st.number_input("Number of Major Vessels (0-3)", 0, 3, 0),
        'thal': st.selectbox("Thalassemia", options=[0, 1, 2]),
    }
    st.session_state.user_input = user_input

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Intro"):
            st.session_state.page = 'intro'
    with col2:
        if st.button("Get Prediction"):
            st.session_state.page = 'prediction'

def prediction_page():
    """Third page: Displays the prediction and custom progress bar."""
    st.title("Prediction Results")
    st.markdown("---")

    # Only run prediction logic once
    if st.session_state.prediction is None:
        user_input = st.session_state.user_input
        if user_input:
            # Preprocessing the User's Inputs
            input_df = pd.DataFrame([user_input])
            
            # Convert selectbox strings to numerical values
            input_df['sex'] = input_df['sex'].map({"Male": 1, "Female": 0})
            input_df['fbs'] = input_df['fbs'].map({"Yes": 1, "No": 0})
            input_df['exang'] = input_df['exang'].map({"Yes": 1, "No": 0})

            # One-hot encode the categorical features
            categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
            
            # Align the input DataFrame to the correct column order
            final_columns = background_data.columns.tolist()
            input_aligned = input_df.reindex(columns=final_columns, fill_value=0)
            
            # Scale the numerical features using the pre-trained scaler
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            input_aligned[numerical_features] = scaler.transform(input_aligned[numerical_features])
            
            st.session_state.input_aligned = input_aligned

            with st.spinner("Making prediction..."):
                time.sleep(1) # Simulate a brief delay for a better UX
                final_input_array = input_aligned.to_numpy()
                prediction = model.predict(final_input_array)[0][0]
                st.session_state.prediction = 'Heart Disease' if prediction > 0.5 else 'No Heart Disease'
                st.session_state.prediction_proba = prediction * 100
    
    # Display the result
    st.subheader(f"Prediction: {st.session_state.prediction}")
    
    # Custom progress bar visualization
    st.markdown("### Risk Level")
    prob = st.session_state.prediction_proba
    if st.session_state.prediction == 'Heart Disease':
        st.write(f"The model predicts a **{prob:.2f}%** risk of heart disease.")
        progress_color = '#d9534f' # Red
    else:
        st.write(f"The model predicts a **{(100-prob):.2f}%** chance of no heart disease.")
        progress_color = '#5cb85c' # Green

    st.markdown(f"""
        <div style="background-color: #e9ecef; border-radius: 20px; height: 30px;">
            <div style="background-color: {progress_color}; height: 100%; width: {prob}%; border-radius: 20px; transition: width 1s;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View SHAP Explanation"):
            st.session_state.page = 'shap_explanation'
    with col2:
        if st.button("Back to Input"):
            st.session_state.page = 'input_form'

def shap_explanation_page():
    """Fourth page: Displays the SHAP force plot with a new explanation."""
    st.title("How the Model Made its Prediction")
    st.markdown("---")

    # Simple explanation of the SHAP chart
    st.markdown("""
        **What does this chart mean?**

        This chart shows how different features of the patient's health data "pushed" the model's prediction. The center line represents the model's average prediction (the "base value").

        * **Red features** on the right side of the line are pushing the prediction **toward** a diagnosis of heart disease.
        * **Blue features** on the left side of the line are pushing the prediction **away** from a diagnosis of heart disease.

        The longer a bar is, the more that feature influenced the final prediction.
    """)

    # Calculate SHAP values only once
    if 'explainer' not in st.session_state or st.session_state.explainer is None:
        st.session_state.explainer = shap.KernelExplainer(model.predict, background_data)
        st.session_state.shap_values = st.session_state.explainer.shap_values(st.session_state.input_aligned)[0].flatten()

    # New code to explain feature contributions in text
    st.markdown("### Feature Contributions")
    
    shap_df = pd.DataFrame({
        'feature': st.session_state.input_aligned.columns,
        'shap_value': st.session_state.shap_values
    })

    # Calculate percentage contribution
    total_abs_shap = shap_df['shap_value'].abs().sum()
    shap_df['contribution'] = (shap_df['shap_value'].abs() / total_abs_shap) * 100

    # Sort by absolute SHAP value to show most influential features first
    shap_df = shap_df.sort_values(by='contribution', ascending=False)
    
    for _, row in shap_df.iterrows():
        feature = row['feature']
        contribution = row['contribution']
        influence = "pushing the prediction **towards heart disease**" if row['shap_value'] > 0 else "pushing the prediction **away from heart disease**"
        
        st.write(f"- **{feature}**: Contributed **{contribution:.2f}%** to the prediction, {influence}.")

    # Display SHAP force plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    shap.force_plot(
        st.session_state.explainer.expected_value[0], 
        st.session_state.shap_values,
        st.session_state.input_aligned.iloc[0], 
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf() # Clear the current figure to prevent it from affecting other plots
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Feature Insights"):
            st.session_state.page = 'insights'
    with col2:
        if st.button("Back to Prediction"):
            st.session_state.page = 'prediction'

def insights_page():
    """Fifth page: Displays feature comparison charts with a dropdown."""
    st.title("Model Insights and Feature Comparison")
    st.markdown("---")
    
    st.write("Explore the data by selecting a feature to see its distribution in patients with and without heart disease.")

    # List of all features from the original dataset
    all_features = df.columns.tolist()
    all_features.remove('target')
    
    selected_feature = st.selectbox("Select a feature to plot", options=all_features)

    st.subheader(f"Distribution of '{selected_feature}' by Heart Disease Status")

    # Check if the selected feature is numerical or categorical to choose the plot type
    if df[selected_feature].dtype in ['int64', 'float64']:
        # Plot for numerical features (Box plot is a good choice for distribution comparison)
        fig = px.box(
            df,
            x='target',
            y=selected_feature,
            color='target',
            title=f"Box Plot of {selected_feature} by Target",
            labels={'target': 'Heart Disease', selected_feature: selected_feature},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Plot for categorical features (Bar chart for counts)
        plot_df = df.groupby(['target', selected_feature]).size().reset_index(name='count')
        fig = px.bar(
            plot_df,
            x=selected_feature,
            y='count',
            color='target',
            barmode='group',
            title=f"Bar Chart of {selected_feature} by Target",
            labels={'target': 'Heart Disease', selected_feature: selected_feature, 'count': 'Count'},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if st.button("Back to SHAP"):
        st.session_state.page = 'shap_explanation'

# --- 5. Main App Flow ---
def main():
    custom_css()
    if st.session_state.page == 'intro':
        intro_page()
    elif st.session_state.page == 'input_form':
        input_form_page()
    elif st.session_state.page == 'prediction':
        prediction_page()
    elif st.session_state.page == 'shap_explanation':
        shap_explanation_page()
    elif st.session_state.page == 'insights':
        insights_page()

if __name__ == "__main__":
    main()
