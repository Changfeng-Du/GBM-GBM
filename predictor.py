import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2e86c1;}
    h2 {color: #28b463;}
    .st-bw {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .prediction-box {background-color: #f8f9f9; padding: 15px; border-radius: 8px; margin: 10px 0;}
    .risk-high {color: #e74c3c; font-weight: bold;}
    .risk-low {color: #28b463; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Load model and data
pmml_model = Model.load('gbm_model.pmml')
dev = pd.read_csv('dev_finally.csv')
vad = pd.read_csv('vad_finally.csv')

feature_names = ['smoker', 'sex', 'carace', 'drink', 'sleep', 'Hypertension', 
                 'Dyslipidemia', 'HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 
                 'LBXWBCSI', 'LBXRBCSI']

# Page configuration
st.set_page_config(page_title="Health Risk Predictor", page_icon="⚕️", layout="wide")

# Main content
st.title("⚕️ Cardiovascular Comorbidity Risk Assessment")
st.markdown("---")

# Input Section
with st.container():
    st.header("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoking Status:", options=[1, 2, 3], 
                            format_func=lambda x: "Never" if x == 1 else "Former" if x == 2 else "Current")
        sex = st.selectbox("Gender:", options=[1, 2], 
                          format_func=lambda x: "Female" if x == 1 else "Male")
        carace = st.selectbox("Ethnicity:", options=[1, 2, 3, 4, 5], 
                            format_func=lambda x: ["Mexican American", "Other Hispanic", 
                                                 "Non-Hispanic White", "Non-Hispanic Black", 
                                                 "Other Race"][x-1])
        drink = st.selectbox("Alcohol Consumption:", options=[1, 2], 
                           format_func=lambda x: "No" if x == 1 else "Yes")
        sleep = st.selectbox("Sleep Disorders:", options=[1, 2], 
                            format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        Hypertension = st.selectbox("Hypertension History:", options=[1, 2], 
                                  format_func=lambda x: "No" if x == 1 else "Yes")
        Dyslipidemia = st.selectbox("Dyslipidemia History:", options=[1, 2], 
                                  format_func=lambda x: "No" if x == 1 else "Yes")
        HHR = st.slider("HHR Ratio:", 0.23, 1.67, 1.0, 0.01)
        RIDAGEYR = st.slider("Age (Years):", 20, 80, 50, 1)
        INDFMPIR = st.slider("Poverty Income Ratio:", 0.0, 5.0, 2.0, 0.1)
        BMXBMI = st.slider("BMI (kg/m²):", 11.5, 67.3, 25.0, 0.1)
        LBXWBCSI = st.slider("White Blood Cells (10⁹/L):", 1.4, 117.2, 6.0, 0.1)
        LBXRBCSI = st.slider("Red Blood Cells (10¹²/L):", 2.52, 7.9, 3.0, 0.01)

# Prediction and Explanation
st.markdown("---")
if st.button("🩺 Assess Risk", use_container_width=True):
    feature_values = [smoker, sex, carace, drink, sleep, Hypertension, 
                     Dyslipidemia, HHR, RIDAGEYR, INDFMPIR, BMXBMI, 
                     LBXWBCSI, LBXRBCSI]
    
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    prediction = pmml_model.predict(input_df)
    prob_0 = prediction['probability(1)'][0]
    prob_1 = prediction['probability(0)'][0]
    predicted_class = 1 if prob_1 > 0.436018256400085 else 0
    probability = prob_1 if predicted_class == 1 else prob_0

    # Display Results
    with st.container():
        st.header("Assessment Results")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h4>Risk Prediction</h4>
                <p>Comorbidity Probability: <span class="{'risk-high' if predicted_class == 1 else 'risk-low'}">{prob_1:.1%}</span></p>
                <p>Non-comorbidity Probability: {prob_0:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:
            st.markdown(f"""
            <div class="prediction-box">
                <h4>Clinical Recommendation</h4>
                {f"<p class='risk-high'>High risk detected - Immediate clinical evaluation recommended</p>" if predicted_class == 1 
                 else "<p class='risk-low'>Low risk detected - Maintain regular monitoring</p>"}
                <small>Threshold: 43.6% probability for comorbidity classification</small>
            </div>
            """, unsafe_allow_html=True)

    # SHAP Explanation
    with st.expander("Feature Impact Analysis (SHAP)", expanded=True):
        background = vad[feature_names].iloc[:100]
        
        def pmml_predict(data):
            if isinstance(data, pd.DataFrame):
                input_df = data[feature_names].copy()
            else:
                input_df = pd.DataFrame(data, columns=feature_names)
            predictions = pmml_model.predict(input_df)
            return np.column_stack((predictions['probability(0)'], predictions['probability(1)']))
        
        explainer = shap.KernelExplainer(pmml_predict, background)
        shap_values = explainer.shap_values(input_df)
        
        plt.figure(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value[1 if predicted_class == 1 else 0],
            shap_values[0, :, 1 if predicted_class == 1 else 0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

    # LIME Explanation
    with st.expander("Local Interpretation (LIME)", expanded=False):
        lime_explainer = LimeTabularExplainer(
            training_data=background.values,
            feature_names=feature_names,
            class_names=['Non-comorbidity', 'Comorbidity'],
            mode='classification'
        )
        
        lime_exp = lime_explainer.explain_instance(
            data_row=input_df.values.flatten(),
            predict_fn=pmml_predict
        )
        
        st.components.v1.html(lime_exp.as_html(show_table=True), height=600)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        Clinical Decision Support System v1.0 · For professional use only
    </div>
    """, unsafe_allow_html=True)
