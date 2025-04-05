import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Comorbidity Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox {margin-bottom: 1.5rem;}
    .highlight {background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem;}
    .risk-meter {background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%); height: 20px;}
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    try:
        model = Model.load('gbm_model.pmml')
        dev = pd.read_csv('dev_finally.csv')
        vad = pd.read_csv('vad_finally.csv')
        return model, dev, vad
    except Exception as e:
        st.error(f"Resource loading failed: {str(e)}")
        st.stop()

pmml_model, dev, vad = load_resources()

# Feature names
FEATURE_NAMES = ['smoker', 'sex','carace', 'drink','sleep','Hypertension', 
                'Dyslipidemia','HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 
                'LBXWBCSI', 'LBXRBCSI']

# Sidebar - User Guide
with st.sidebar:
    st.header("User Guide")
    st.markdown("""
    1. Fill all health parameters
    2. Click Predict button for results
    3. Review risk explanations
    4. See tooltips for parameter details
    """)
    st.divider()
    st.caption("Data Source: NHANES Database | Model Version: GBM v2.1")

# Main interface
st.title("❤️ Cardiovascular Comorbidity Risk Prediction System")
st.markdown("Machine learning-based risk assessment with SHAP/LIME explanations")

# Input panel
with st.expander("🖋️ Health Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Info")
        smoker = st.selectbox("Smoking Status", options=[1, 2, 3], 
                            format_func=lambda x: "Never" if x == 1 else "Former" if x == 2 else "Current",
                            help="Includes both traditional and e-cigarettes")
        sex = st.selectbox("Gender", options=[1, 2], 
                         format_func=lambda x: "Female" if x == 1 else "Male")
        carace = st.selectbox("Ethnicity", options=[1, 2, 3, 4, 5], 
                            format_func=lambda x: "Mexican American" if x == 1 else "Other Hispanic" if x == 2 
                            else "Non-Hispanic White" if x == 3 else "Non-Hispanic Black" if x == 4 else "Other")
        
    with col2:
        st.subheader("Lifestyle")
        drink = st.selectbox("Alcohol Use", options=[1, 2], 
                           format_func=lambda x: "No" if x == 1 else "Yes",
                           help="Alcohol consumption in past 12 months")
        sleep = st.selectbox("Sleep Issues", options=[1, 2], 
                           format_func=lambda x: "Yes" if x == 1 else "No")
        BMXBMI = st.number_input("BMI (kg/m²)", min_value=11.5, max_value=67.3, value=25.0,
                               help="Normal range: 18.5-24.9")
        
    with col3:
        st.subheader("Clinical Metrics")
        HHR = st.number_input("HHR Ratio", min_value=0.23, max_value=1.67, value=1.0,
                            help="Heart rate recovery ratio (Normal: 0-1)")
        RIDAGEYR = st.number_input("Age (years)", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Income Ratio", min_value=0.0, max_value=5.0, value=2.0,
                                 help="0: Lowest income, 5: Highest income")

# Prediction and visualization
if st.button("🔍 Run Prediction", type="primary", use_container_width=True):
    feature_values = [smoker, sex, carace, drink, sleep, 
                     Hypertension, Dyslipidemia, HHR, RIDAGEYR, 
                     INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI]
    
    with st.spinner("Analyzing..."):
        input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
        prediction = pmml_model.predict(input_df)
        
        # Results display
        prob_1 = prediction['probability(1)'][0]
        prob_0 = prediction['probability(0)'][0]
        risk_level = "High Risk" if prob_1 > 0.436 else "Low Risk"
        
        # Risk visualization
        st.subheader(f"Prediction Result: {risk_level}")
        risk_color = "#dc3545" if risk_level == "High Risk" else "#28a745"
        st.markdown(f"""
        <div class="highlight">
            <div style="display: flex; justify-content: space-between; align-items: center">
                <div>
                    <h4 style="color:{risk_color}; margin:0">Comorbidity Probability: {prob_1*100:.1f}%</h4>
                    <p>Non-comorbidity Probability: {prob_0*100:.1f}%</p>
                </div>
                <div style="width: 200px">
                    <div class="risk-meter" style="border-radius:10px; margin-bottom:8px"></div>
                    <div style="text-align:center">Risk Indicator</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Medical advice
        if risk_level == "High Risk":
            advice = f"""
            <div style="color:#dc3545; margin:1rem 0">
            📢 Recommendation: Elevated comorbidity risk detected ({prob_1*100:.1f}%), suggested actions:
            <ul>
                <li>Consult a cardiologist immediately</li>
                <li>Monitor blood pressure and lipid profile</li>
                <li>Improve lifestyle (diet & exercise)</li>
            </ul>
            </div>
            """
        else:
            advice = f"""
            <div style="color:#28a745; margin:1rem 0">
            ✅ Current risk manageable ({prob_1*100:.1f}%), suggested actions:
            <ul>
                <li>Regular health checkups (bi-annual recommended)</li>
                <li>Maintain healthy BMI (<25)</li>
                <li>Monitor sleep quality and stress</li>
            </ul>
            </div>
            """
        st.markdown(advice, unsafe_allow_html=True)
        
        # Explanation tabs
        tab1, tab2 = st.tabs(["📈 SHAP Analysis", "🔎 LIME Explanation"])
        
        with tab1:
            plt.figure(figsize=(10, 6))
            background = vad[FEATURE_NAMES].iloc[:100]
            
            def pmml_predict(data):
                return np.column_stack((
                    pmml_model.predict(data)['probability(0)'],
                    pmml_model.predict(data)['probability(1)']
                ))
            
            explainer = shap.KernelExplainer(pmml_predict, background)
            shap_values = explainer.shap_values(input_df)
            
            if risk_level == "High Risk":
                shap.force_plot(explainer.expected_value[1], 
                              shap_values[0][:,1], 
                              input_df.iloc[0],
                              matplotlib=True,
                              show=False,
                              text_rotation=15)
            else:
                shap.force_plot(explainer.expected_value[0], 
                              shap_values[0][:,0], 
                              input_df.iloc[0],
                              matplotlib=True,
                              show=False,
                              text_rotation=15)
            
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("SHAP force plot: Red features increase risk, blue features decrease risk")
        
        with tab2:
            lime_explainer = LimeTabularExplainer(
                training_data=background.values,
                feature_names=FEATURE_NAMES,
                class_names=['Non-comorbidity', 'Comorbidity'],
                mode='classification',
                verbose=False
            )
            
            lime_exp = lime_explainer.explain_instance(
                input_df.values.flatten(), 
                pmml_predict,
                num_features=10
            )
            
            st.components.v1.html(lime_exp.as_html(show_table=True), 
                                height=600, 
                                scrolling=True)

st.divider()
st.markdown("⚠️ Disclaimer: Prediction results are for reference only, not a substitute for professional medical diagnosis")
