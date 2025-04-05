import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model
from matplotlib import rcParams

# Set global font size for better readability
rcParams['font.size'] = 12

# Load the PMML model
pmml_model = Model.load('gbm_model.pmml')
# Load the data
dev = pd.read_csv('dev_finally.csv')
vad = pd.read_csv('vad_finally.csv')

# Define feature names in the correct order (from PMML model)
feature_names = ['smoker', 'sex','carace', 'drink','sleep','Hypertension', 'Dyslipidemia','HHR', 'RIDAGEYR', 
                 'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI']

# Streamlit user interface
st.title("Co-occurrence of Myocardial Infarction and Stroke Predictor")

# Create input columns to organize widgets better
col1, col2 = st.columns(2)

with col1:
    smoker = st.selectbox("Smoker:", options=[1, 2, 3], 
                         format_func=lambda x: "Never" if x == 1 else "Former" if x == 2 else "Current")
    sex = st.selectbox("Sex:", options=[1, 2], 
                       format_func=lambda x: "Female" if x == 1 else "Male")
    carace = st.selectbox("Race/Ethnicity:", options=[1, 2, 3, 4, 5], 
                         format_func=lambda x: "Mexican American" if x == 1 else "Other Hispanic" if x == 2 
                         else "Non-Hispanic White" if x == 3 else "Non-Hispanic Black" if x == 4 else "Other Race")
    drink = st.selectbox("Alcohol Consumption:", options=[1, 2], 
                        format_func=lambda x: "No" if x == 1 else "Yes")
    sleep = st.selectbox("Sleep Problem:", options=[1, 2], 
                         format_func=lambda x: "Yes" if x == 1 else "No")
    Hypertension = st.selectbox("Hypertension:", options=[1, 2], 
                                format_func=lambda x: "No" if x == 1 else "Yes")
    Dyslipidemia = st.selectbox("Dyslipidemia:", options=[1, 2], 
                                format_func=lambda x: "No" if x == 1 else "Yes")

with col2:
    HHR = st.number_input("HHR Ratio:", min_value=0.23, max_value=1.67, value=1.0)
    RIDAGEYR = st.number_input("Age (years):", min_value=20, max_value=80, value=50)
    INDFMPIR = st.number_input("Poverty Income Ratio:", min_value=0.0, max_value=5.0, value=2.0)
    BMXBMI = st.number_input("Body Mass Index (kg/m²):", min_value=11.5, max_value=67.3, value=25.0)
    LBXWBCSI = st.number_input("White Blood Cell Count (10^9/L):", min_value=1.4, max_value=117.2, value=6.0)
    LBXRBCSI = st.number_input("Red Blood Cell Count (10^9/L):", min_value=2.52, max_value=7.9, value=3.0)

# Process inputs and make predictions
feature_values = [smoker, sex, carace, drink, sleep, Hypertension, Dyslipidemia, HHR, RIDAGEYR, 
                 INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI]

if st.button("Predict"):
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Make prediction
    prediction = pmml_model.predict(input_df)
    prob_0 = prediction['probability(1)'][0]
    prob_1 = prediction['probability(0)'][0]
    
    # Determine predicted class
    predicted_class = 1 if prob_1 > 0.436018256400085 else 0
    probability = prob_1 if predicted_class == 1 else prob_0
    
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Comorbidity, 0: Non-comorbidity)")
    st.write(f"**Probability of Comorbidity:** {prob_1:.4f}")
    st.write(f"**Probability of Non-comorbidity:** {prob_0:.4f}")

    # Generate advice
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of co-occurrence of myocardial infarction and stroke disease. "
            f"The model predicts a {probability*100:.1f}% probability. "
            "It's advised to consult with your healthcare provider for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk ({(1-probability)*100:.1f}% probability). "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups."
        )
    st.write(advice)

    # SHAP Explanation
    st.subheader("Model Explanation with SHAP")
    st.write("SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction.")
    
    # Prepare background data (using first 100 samples)
    background = vad[feature_names].iloc[:100]
    
    # Define prediction function for SHAP
    def pmml_predict(data):
        if isinstance(data, pd.DataFrame):
            input_df = data[feature_names].copy()
        else:
            input_df = pd.DataFrame(data, columns=feature_names)
        
        predictions = pmml_model.predict(input_df)
        return np.column_stack((predictions['probability(0)'], predictions['probability(1)']))
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(pmml_predict, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_df)
    
    # Display SHAP force plot with improved formatting
    st.subheader("SHAP Force Plot")
    st.write("This plot shows how each feature pushes the prediction from the base value (average model output) to the final prediction.")
    
    plt.figure(figsize=(12, 6), dpi=100)
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0,:,1],  # Take SHAP values for class 1
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False,
                       text_rotation=15,
                       figsize=(12, 6))
    else:
        shap.force_plot(explainer.expected_value[0], 
                       shap_values[0,:,0],  # Take SHAP values for class 0
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False,
                       text_rotation=15,
                       figsize=(12, 6))
    
    plt.tight_layout()
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()
    
    # Display SHAP summary plot
    st.subheader("SHAP Summary Plot")
    st.write("This plot shows the importance of features and their impact on predictions.")
    
    plt.figure(figsize=(12, 8), dpi=100)
    shap.summary_plot(shap_values[predicted_class], 
                     input_df, 
                     plot_type="bar",
                     max_display=len(feature_names),
                     show=False)
    plt.title("Feature Importance", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()
    
    # Display SHAP detailed plot
    st.subheader("Detailed SHAP Values")
    st.write("This plot shows how each feature's value affects the prediction.")
    
    plt.figure(figsize=(12, 8), dpi=100)
    shap.summary_plot(shap_values[predicted_class], 
                     input_df, 
                     plot_type="dot",
                     max_display=len(feature_names),
                     show=False)
    plt.title("Feature Impact on Prediction", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf(), bbox_inches='tight')
    plt.clf()

    # LIME Explanation
    st.subheader("Model Explanation with LIME")
    st.write("LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the model locally.")
    
    lime_explainer = LimeTabularExplainer(
        training_data=background.values,
        feature_names=feature_names,
        class_names=['Non-comorbidity', 'Comorbidity'],
        mode='classification',
        discretize_continuous=False,
        kernel_width=3,
        verbose=True
    )
    
    lime_exp = lime_explainer.explain_instance(
        data_row=input_df.values.flatten(),
        predict_fn=pmml_predict,
        num_features=len(feature_names),
        top_labels=1
    )
    
    # Display LIME explanation with improved formatting
    st.subheader("LIME Explanation Plot")
    
    # Create a larger figure for the LIME plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    lime_exp.as_pyplot_figure(label=predicted_class)
    plt.title("LIME Explanation", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    
    # Display LIME explanation as HTML with custom styling
    st.subheader("Detailed LIME Explanation")
    lime_html = lime_exp.as_html(predict_proba=True, show_predicted_value=True)
    
    # Add custom CSS for better display
    lime_html = f"""
    <style>
        .lime {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }}
        .lime .explanation {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .lime .features {{
            margin-top: 15px;
        }}
        .lime .feature {{
            margin-bottom: 8px;
            padding: 8px;
            background-color: white;
            border-left: 4px solid #4CAF50;
        }}
        .lime .feature.pro {{
            border-left-color: #4CAF50;
        }}
        .lime .feature.con {{
            border-left-color: #F44336;
        }}
        .lime .feature.weight {{
            font-weight: bold;
        }}
    </style>
    {lime_html}
    """
    
    st.components.v1.html(lime_html, height=1000, scrolling=True)
