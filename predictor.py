import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_resources():
    model = Model.load('gbm_model.pmml')
    dev = pd.read_csv('dev_finally.csv')
    vad = pd.read_csv('vad_finally.csv')
    return model, dev, vad

pmml_model, dev, vad = load_resources()

# Feature configuration
feature_names = ['smoker', 'sex', 'carace', 'drink', 'sleep', 'Hypertension',
                'Dyslipidemia', 'HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI',
                'LBXWBCSI', 'LBXRBCSI']

# Sidebar inputs
st.sidebar.header("Patient Information Input")
with st.sidebar.expander("Basic Information", expanded=True):
    sex = st.selectbox("Sex", [1, 2], 
                     format_func=lambda x: "Female" if x == 1 else "Male")
    RIDAGEYR = st.number_input("Age (years)", 20, 80, 50)
    carace = st.selectbox("Race/Ethnicity", [1, 2, 3, 4, 5],
                        format_func=lambda x: ["Mexican American", "Other Hispanic",
                                              "Non-Hispanic White", "Non-Hispanic Black",
                                              "Other Race"][x-1])

with st.sidebar.expander("Lifestyle Factors", expanded=True):
    smoker = st.selectbox("Smoking Status", [1, 2, 3],
                        format_func=lambda x: ["Never", "Former", "Current"][x-1])
    drink = st.selectbox("Alcohol Consumption", [1, 2],
                       format_func=lambda x: "No" if x == 1 else "Yes")
    sleep = st.selectbox("Sleep Quality", [1, 2],
                       format_func=lambda x: "Sleep Problems" if x == 1 else "Normal")

with st.sidebar.expander("Health Metrics"):
    Hypertension = st.selectbox("Hypertension", [1, 2],
                              format_func=lambda x: "No" if x == 1 else "Yes")
    Dyslipidemia = st.selectbox("Dyslipidemia", [1, 2],
                              format_func=lambda x: "No" if x == 1 else "Yes")
    HHR = st.number_input("Waist-to-Hip Ratio", 0.23, 1.67, 1.0,
                        help="Normal range: 0.7-1.0")
    BMXBMI = st.number_input("BMI (kg/m²)", 11.5, 67.3, 25.0)
    INDFMPIR = st.number_input("Poverty Income Ratio", 0.0, 5.0, 2.0,
                             help="0 = Lowest income, 5 = Highest income")

with st.sidebar.expander("Blood Indicators"):
    LBXWBCSI = st.number_input("White Blood Cells (10^9/L)", 1.4, 117.2, 6.0)
    LBXRBCSI = st.number_input("Red Blood Cells (10^9/L)", 2.52, 7.9, 3.0)

# Main interface
st.title("❤️ Cardiovascular Comorbidity Risk Prediction")
st.markdown("---")

if st.sidebar.button("Predict", type="primary"):
    # Data processing
    feature_values = [smoker, sex, carace, drink, sleep, Hypertension,
                     Dyslipidemia, HHR, RIDAGEYR, INDFMPIR, BMXBMI,
                     LBXWBCSI, LBXRBCSI]
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Prediction
    prediction = pmml_model.predict(input_df)
    prob_0 = prediction['probability(1)'][0]
    prob_1 = prediction['probability(0)'][0]
    predicted_class = 1 if prob_1 > 0.436 else 0
    
    # Results display
    main_col1, main_col2 = st.columns([1, 2])
    
    with main_col1:
        st.subheader("Prediction Result")
        risk_color = "#ff4b4b" if predicted_class == 1 else "#2ecc71"
        st.markdown(f"""
        <div style="border:2px solid {risk_color}; border-radius:10px; padding:20px">
            <h3 style="color:{risk_color}; margin:0">{"High Risk" if predicted_class == 1 else "Low Risk"}</h3>
            <p style="font-size:1.2em">Comorbidity Probability: <b>{prob_1*100:.1f}%</b></p>
            <p style="font-size:1.2em">Non-comorbidity Probability: <b>{prob_0*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health recommendations
        st.subheader("Health Recommendations")
        if predicted_class == 1:
            advice = [
                "Recommend scheduling a cardiology consultation",
                "Maintain a balanced diet and regular exercise",
                "Monitor blood pressure and lipid levels regularly",
                "Avoid smoking and excessive alcohol consumption"
            ]
        else:
            advice = [
                "Continue maintaining healthy lifestyle habits",
                "Recommend annual health check-ups",
                "Maintain healthy weight range",
                "Manage stress and mental health"
            ]
        for item in advice:
            st.markdown(f"- {item}")

    with main_col2:
        # SHAP explanation
        st.subheader("Feature Impact Analysis (SHAP)")
        background = vad[feature_names].iloc[:100]
        
        def pmml_predict(data):
            return np.column_stack((
                pmml_model.predict(data)['probability(0)'],
                pmml_model.predict(data)['probability(1)']
            ))
        
        explainer = shap.KernelExplainer(pmml_predict, background)
        shap_values = explainer.shap_values(input_df)
        
        plt.figure(figsize=(10, 6))
        shap_class = 1 if predicted_class == 1 else 0
        shap.force_plot(
            explainer.expected_value[shap_class],
            shap_values[0][:, shap_class],
            input_df.iloc[0],
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        st.pyplot(plt, use_container_width=True)
        plt.clf()

        # LIME explanation
        st.subheader("Local Explanation (LIME)")
        lime_explainer = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['Non-comorbidity', 'Comorbidity'],
            verbose=True,
            mode='classification'
        )
        
        lime_exp = lime_explainer.explain_instance(
            input_df.values.flatten(),
            pmml_predict,
            num_features=10
        )
        
        st.components.v1.html(lime_exp.as_html(show_table=True), height=800)

else:
    st.info("Please input patient information and click [Predict] button")
    st.image("healthcare_banner.jpg", use_column_width=True)

st.markdown("---")
st.caption("Note: This prediction is for reference only and cannot replace professional medical diagnosis.")
