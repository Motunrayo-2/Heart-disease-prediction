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
    st.image('image.jpeg', use_column_width=True)


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

 # ------------------------------------------------------------------
# NEW  shap_explanation_page  (drop-in replacement)
# ------------------------------------------------------------------
def shap_explanation_page():
    st.title("How the Model Made its Prediction")
    st.markdown("---")

    # 1. Use ONLY the custom input that was already stored
    X_row = st.session_state.input_aligned

    # 2. Build / reuse the SHAP explainer (cached once)
    if "explainer" not in st.session_state or st.session_state.explainer is None:
        st.session_state.explainer = shap.KernelExplainer(
            model.predict, background_data
        )

    shap_vals = st.session_state.explainer.shap_values(X_row)[0].flatten()

    # 3. Text contributions
    st.markdown("### Feature Contributions")
    shap_df = pd.DataFrame({
        "feature": X_row.columns,
        "shap_value": shap_vals
    })
    total = shap_df["shap_value"].abs().sum()
    shap_df["contrib_pct"] = shap_df["shap_value"].abs() / total * 100
    shap_df = shap_df.sort_values("contrib_pct", ascending=False)

    for _, r in shap_df.head(10).iterrows():
        direction = "toward" if r.shap_value > 0 else "away from"
        st.write(
            f"- **{r.feature}** contributed **{r.contrib_pct:.1f}%**, pushing the model {direction} disease"
        )

    # 4. Force plot – fixed call
fig = shap.force_plot(
    st.session_state.explainer.expected_value,
    shap_vals,
    X_row.iloc[0:1],          # keep column names
    matplotlib=True,
    show=False
)
# shap returns a matplotlib Figure; display it
st.pyplot(fig)
plt.clf()

    # 5. Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Feature Insights"):
            st.session_state.page = "insights"
    with col2:
        if st.button("Back to Prediction"):
            st.session_state.page = "prediction"
# ------------------------------------------------------------------
# NEW insights_page – with two download buttons
# ------------------------------------------------------------------
def insights_page():
    st.title("Model Insights and Feature Comparison")
    st.markdown("---")

    # ------------------------------------------------------------------
    # 1. Sidebar controls (unchanged)
    # ------------------------------------------------------------------
    st.sidebar.subheader("Filters")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols.remove("target")
    selected_feat = st.sidebar.selectbox("Select a feature to analyse", numeric_cols)

    show_scatter = st.sidebar.checkbox("Compare two features (scatter)")
    if show_scatter:
        second_feat = st.sidebar.selectbox("Second feature", [c for c in numeric_cols if c != selected_feat])

    # ------------------------------------------------------------------
    # 2. Univariate plot + stats (unchanged)
    # ------------------------------------------------------------------
    st.subheader(f"Distribution of '{selected_feat}' by Heart Disease status")
    plot_type = st.radio("Plot style", ["box", "violin"], horizontal=True)
    if plot_type == "box":
        fig = px.box(df, x="target", y=selected_feat, color="target",
                     labels={"target": "Heart Disease"}, height=350)
    else:
        fig = px.violin(df, x="target", y=selected_feat, color="target",
                        box=True, labels={"target": "Heart Disease"}, height=350)
    st.plotly_chart(fig, use_container_width=True)

    from scipy import stats
    grp0 = df[df["target"] == 0][selected_feat]
    grp1 = df[df["target"] == 1][selected_feat]
    t_stat, p_val = stats.ttest_ind(grp0, grp1, equal_var=False)
    st.info(
        f"**Welch’s t-test** for {selected_feat}:  t = {t_stat:.2f},  p = {p_val:.4f}"
        + (" ➜ **significant**" if p_val < 0.05 else " ➜ **not significant**")
    )

    st.write("**Descriptive statistics**")
    summary = df.groupby("target")[selected_feat].describe().T
    st.dataframe(summary.style.format("{:.2f}"))

    # ------------------------------------------------------------------
    # 3. Optional bivariate scatter (unchanged)
    # ------------------------------------------------------------------
    if show_scatter:
        st.subheader(f"Scatter: {selected_feat} vs {second_feat}")
        fig2 = px.scatter(
            df,
            x=selected_feat,
            y=second_feat,
            color="target",
            trendline="ols",
            labels={"target": "Heart Disease"},
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Correlation heat-map (unchanged)
    # ------------------------------------------------------------------
    st.subheader("Correlation heat-map (numeric features)")
    corr = df[numeric_cols + ["target"]].corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. DOWNLOAD SECTION
    # ------------------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns(2)

    # 5a. Download the PUBLIC dataset
    public_csv = df.drop(columns=["label"], errors="ignore").to_csv(index=False)
    with col1:
        st.download_button(
            label="📁 Download Public Training Data",
            data=public_csv,
            file_name="public_heart_data.csv",
            mime="text/csv"
        )

    # 5b. Download the USER’S INPUT row
    if st.session_state.user_input is not None:
        # Re-assemble the row exactly as the user typed it
        user_dict = st.session_state.user_input.copy()
        # Add the prediction and probability
        user_dict["predicted_class"] = st.session_state.prediction
        user_dict["predicted_probability_%"] = round(st.session_state.prediction_proba, 2)

        user_df = pd.DataFrame([user_dict])
        user_csv = user_df.to_csv(index=False)
        with col2:
            st.download_button(
                label="📥 Download My Input Row",
                data=user_csv,
                file_name="my_patient_input.csv",
                mime="text/csv"
            )
    else:
        with col2:
            st.write("No custom input yet.")

    # ------------------------------------------------------------------
    # 6. Navigation
    # ------------------------------------------------------------------
    st.markdown("---")
    if st.button("Back to SHAP"):
        st.session_state.page = "shap_explanation"

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
