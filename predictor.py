import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# 设置页面配置
st.set_page_config(
    page_title="Myocardial Infarction and Stroke Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载 PMML 模型
pmml_model = Model.load('gbm_model.pmml')
# 加载数据
dev = pd.read_csv('dev_finally.csv')
vad = pd.read_csv('vad_finally.csv')

# 定义特征名称
feature_names = ['smoker', 'sex', 'carace', 'drink', 'sleep', 'Hypertension', 
                 'Dyslipidemia', 'HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 
                 'LBXWBCSI', 'LBXRBCSI']

# 自定义样式
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<h1 class="big-font">Co-occurrence of Myocardial Infarction and Stroke Predictor</h1>', unsafe_allow_html=True)

# 创建输入列
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

with col2:
    Hypertension = st.selectbox("Hypertension:", options=[1, 2], 
                                format_func=lambda x: "No" if x == 1 else "Yes")
    Dyslipidemia = st.selectbox("Dyslipidemia:", options=[1, 2], 
                                format_func=lambda x: "No" if x == 1 else "Yes")
    HHR = st.number_input("HHR Ratio:", min_value=0.23, max_value=1.67, value=1.0)
    RIDAGEYR = st.number_input("Age (years):", min_value=20, max_value=80, value=50)
    INDFMPIR = st.number_input("Poverty Income Ratio:", min_value=0.0, max_value=5.0, value=2.0)
    BMXBMI = st.number_input("Body Mass Index (kg/m²):", min_value=11.5, max_value=67.3, value=25.0)
    LBXWBCSI = st.number_input("White Blood Cell Count (10^9/L):", min_value=1.4, max_value=117.2, value=6.0)
    LBXRBCSI = st.number_input("Red Blood Cell Count (10^9/L):", min_value=2.52, max_value=7.9, value=3.0)

# 处理输入并进行预测
feature_values = [smoker, sex, carace, drink, sleep, Hypertension, Dyslipidemia, HHR, RIDAGEYR, 
                 INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI]

if st.button("Predict"):
    # 创建 DataFrame
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # 进行预测
    prediction = pmml_model.predict(input_df)
    prob_0 = prediction['probability(1)'][0]
    prob_1 = prediction['probability(0)'][0]
    
    # 确定预测类别
    predicted_class = 1 if prob_1 > 0.436018256400085 else 0
    probability = prob_1 if predicted_class == 1 else prob_0
    
    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Comorbidity, 0: Non-comorbidity)")
    st.write(f"**Probability of Comorbidity:** {prob_1:.4f}")
    st.write(f"**Probability of Non-comorbidity:** {prob_0:.4f}")

    # 生成建议
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

    # SHAP 解释
    st.subheader("SHAP Explanation")
    
    # 准备背景数据
    background = vad[feature_names].iloc[:100]
    
    # 定义 SHAP 预测函数
    def pmml_predict(data):
        if isinstance(data, pd.DataFrame):
            input_df = data[feature_names].copy()
        else:
            input_df = pd.DataFrame(data, columns=feature_names)
        
        predictions = pmml_model.predict(input_df)
        return np.column_stack((predictions['probability(0)'], predictions['probability(1)']))
    
    # 创建 SHAP 解释器
    explainer = shap.KernelExplainer(pmml_predict, background)
    
    # 计算 SHAP 值
    shap_values = explainer.shap_values(input_df)
    
    # 显示 SHAP 力量图
    st.subheader("SHAP Force Plot Explanation")
    plt.figure()
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], 
                         shap_values[0, :, 1],  # 取类 1 的 SHAP 值
                         input_df.iloc[0],
                         matplotlib=True,
                         show=False)
    else:
        shap.force_plot(explainer.expected_value[0], 
                         shap_values[0, :, 0],  # 取类 0 的 SHAP 值
                         input_df.iloc[0],
                         matplotlib=True,
                         show=False)
    
    st.pyplot(plt.gcf())
    plt.clf()

    # LIME 解释
    st.subheader("LIME Explanation")
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
    
    # 显示 LIME 解释
    lime_html = lime_exp.as_html(show_table=True)  
    st.components.v1.html(lime_html, height=800, scrolling=True)
