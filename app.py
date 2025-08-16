import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import time

# ---------- load assets ----------
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('MY_ANN_model.h5')
        scaler = joblib.load('scaler.joblib')
        background_data = pd.read_csv('background_data.csv')
        df = pd.read_csv('heart.csv')
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure all necessary files are in the app's directory.")
        st.stop()
    return model, scaler, background_data, df

model, scaler, background_data, df = load_assets()

# ---------- session state ----------
for key in ['page', 'user_input', 'prediction', 'prediction_proba', 'input_aligned', 'explainer', 'shap_vals']:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- page functions ----------
def intro_page():
    st.title("Cardiovascular Health: Heart Disease Risk Prediction")
    st.markdown("---")
    st.write("Welcome! This app estimates heart-disease risk from patient metrics.")
    st.image('image.jpeg', use_column_width=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Start Inputting Features", key="intro_next", on_click=lambda: setattr(st.session_state, "page", "input_form"),
                  type="primary", help="", use_container_width=True)

def input_form_page():
    st.title("Patient's Health Metrics")
    st.markdown("---")
    user_input = {
        'age': st.slider("Age", 20, 100, 50),
        'sex': st.selectbox("Sex", ["Male", "Female"]),
        'cp': st.selectbox("Chest Pain Type", [0, 1, 2, 3]),
        'trestbps': st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120),
        'chol': st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200),
        'fbs': st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"]),
        'restecg': st.selectbox("Resting ECG Results", [0, 1, 2]),
        'thalach': st.slider("Maximum Heart Rate Achieved", 70, 220, 150),
        'exang': st.selectbox("Exercise Induced Angina", ["No", "Yes"]),
        'oldpeak': st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1),
        'slope': st.selectbox("Slope of Peak Exercise ST", [0, 1, 2]),
        'ca': st.number_input("Number of Major Vessels (0-3)", 0, 3, 0),
        'thal': st.selectbox("Thalassemia", [0, 1, 2]),
    }
    st.session_state.user_input = user_input
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Intro", key="input_back", on_click=lambda: setattr(st.session_state, "page", "intro"),
                  type="secondary", help="", use_container_width=True)
    with col2:
        st.button("Get Prediction", key="input_next", on_click=lambda: setattr(st.session_state, "page", "prediction"),
                  type="primary", help="", use_container_width=True)

def prediction_page():
    st.title("Prediction Results")
    st.markdown("---")
    if st.session_state.prediction is None:
        user_input = st.session_state.user_input
        if user_input:
            input_df = pd.DataFrame([user_input])
            input_df['sex'] = input_df['sex'].map({"Male": 1, "Female": 0})
            input_df['fbs'] = input_df['fbs'].map({"Yes": 1, "No": 0})
            input_df['exang'] = input_df['exang'].map({"Yes": 1, "No": 0})
            cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
            input_df = input_df.reindex(columns=background_data.columns, fill_value=0)
            num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            st.session_state.input_aligned = input_df
            with st.spinner("Making prediction..."):
                pred = model.predict(input_df.to_numpy())[0][0]
                st.session_state.prediction = 'Heart Disease' if pred > 0.5 else 'No Heart Disease'
                st.session_state.prediction_proba = pred * 100

    prob = st.session_state.prediction_proba
    st.subheader(f"Prediction: {st.session_state.prediction}")
    colour = "#e74c3c" if st.session_state.prediction == 'Heart Disease' else "#28a745"
    st.markdown("### Risk Level")
    st.write(
        f"The model predicts a **{prob:.1f}%** risk of heart disease."
        if st.session_state.prediction == 'Heart Disease'
        else f"The model predicts a **{(100-prob):.1f}%** chance of no heart disease."
    )
    st.markdown(f"""
        <div style="background-color: #e9ecef; border-radius: 20px; height: 30px;">
            <div style="background-color: {colour}; height: 100%; width: {prob}%; border-radius: 20px;"></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Input", key="pred_back", on_click=lambda: setattr(st.session_state, "page", "input_form"),
                  type="secondary", help="", use_container_width=True)
    with col2:
        st.button("View SHAP Explanation", key="pred_next", on_click=lambda: setattr(st.session_state, "page", "shap_explanation"),
                  type="primary", help="", use_container_width=True)

def shap_explanation_page():
    st.title("How the Model Made its Prediction")
    st.markdown("---")
    if st.session_state.input_aligned is None or st.session_state.input_aligned.empty:
        st.warning("No patient data available.")
        st.button("Back to Prediction", on_click=lambda: setattr(st.session_state, "page", "prediction"))
        return

    X_row = st.session_state.input_aligned
    if "explainer" not in st.session_state or st.session_state.explainer is None:
        st.session_state.explainer = shap.KernelExplainer(model.predict, background_data)
    if "shap_vals" not in st.session_state or st.session_state.shap_vals is None:
        st.session_state.shap_vals = st.session_state.explainer.shap_values(X_row)[0].flatten()

    shap_vals = st.session_state.shap_vals
    shap_df = pd.DataFrame({"feature": X_row.columns, "value": X_row.iloc[0], "shap": shap_vals})
    shap_df["abs_shap"] = shap_df["shap"].abs()
    shap_df = shap_df.sort_values("abs_shap")

    fig, ax = plt.subplots(figsize=(5, max(4, len(shap_df) * 0.3)))
    colors = ["#3498db" if s < 0 else "#e74c3c" for s in shap_df["shap"]]
    ax.barh(shap_df["feature"], shap_df["shap"], color=colors)
    ax.set_xlabel("SHAP value")
    ax.axvline(0, color="black")
    plt.tight_layout()
    st.pyplot(fig)

    total = shap_df["abs_shap"].sum()
    shap_df["contrib_pct"] = shap_df["abs_shap"] / total * 100
    st.dataframe(
        shap_df[["feature", "value", "shap", "contrib_pct"]]
        .sort_values("contrib_pct", ascending=False)
        .style.format({"value": "{:.2f}", "shap": "{:.3f}", "contrib_pct": "{:.1f}%"})
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Prediction", key="shap_back", on_click=lambda: setattr(st.session_state, "page", "prediction"),
                  type="secondary", help="", use_container_width=True)
    with col2:
        st.button("View Feature Insights", key="shap_next", on_click=lambda: setattr(st.session_state, "page", "insights"),
                  type="primary", help="", use_container_width=True)

def insights_page():
    st.title("Model Insights and Feature Comparison")
    st.markdown("---")

    # ---------- Sidebar controls ----------
    st.sidebar.subheader("Controls")
    numeric_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "target"]
    selected_feat = st.sidebar.selectbox("Pick a feature", numeric_cols)

    show_scatter = st.sidebar.checkbox("Compare two features")
    second_feat = None
    if show_scatter:
        second_feat = st.sidebar.selectbox("Second feature", [c for c in numeric_cols if c != selected_feat])

    # ---------- 1️⃣  Univariate ----------
    st.subheader(f"Distribution of '{selected_feat}' by Heart Disease")
    plot_type = st.radio("Plot style", ["box", "violin"], horizontal=True)
    fig = (px.box(df, x="target", y=selected_feat, color="target")
           if plot_type == "box"
           else px.violin(df, x="target", y=selected_feat, color="target", box=True))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- 2️⃣  Scatter comparison ----------
    if show_scatter and second_feat:
        st.subheader(f"Scatter: {selected_feat} vs {second_feat}")
        fig2 = px.scatter(df, x=selected_feat, y=second_feat, color="target")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- 3️⃣  Downloads ----------
    st.markdown("---")
    col1, col2 = st.columns(2)

    # Public data
    public_csv = df.to_csv(index=False)
    col1.download_button("📁 Download Public Training Data", public_csv,
                         file_name="public_heart_data.csv", mime="text/csv")

    # User row
    if st.session_state.user_input is not None:
        user_dict = st.session_state.user_input.copy()
        user_dict["predicted_class"] = st.session_state.prediction
        user_dict["predicted_probability_%"] = round(st.session_state.prediction_proba, 2)
        user_csv = pd.DataFrame([user_dict]).to_csv(index=False)
        col2.download_button("📥 Download My Input Row", user_csv,
                             file_name="my_patient_input.csv", mime="text/csv")
    else:
        col2.write("No custom input yet.")

    # ---------- 4️⃣  Navigation ----------
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        nav_button("Back to SHAP", "shap_explanation", bg="#dc3545")
    with col2:
        nav_button("🎉 Finish & Start Over", "intro", bg="#28a745")

def main():
    page = st.session_state.page
    if page == 'intro':
        intro_page()
    elif page == 'input_form':
        input_form_page()
    elif page == 'prediction':
        prediction_page()
    elif page == 'shap_explanation':
        shap_explanation_page()
    elif page == 'insights':
        insights_page()

if __name__ == "__main__":
    main()
