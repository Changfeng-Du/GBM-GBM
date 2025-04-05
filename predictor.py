import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pypmml import Model

# 页面配置
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .result-box { padding:20px; border-radius:10px; margin:20px 0; }
    .high-risk { background-color: #ffd4d4; border-left:5px solid #ff0000; }
    .low-risk { background-color: #d6f5d6; border-left:5px solid #00aa00; }
    .feature-label { font-weight:600; color:#2c3e50; }
    .stNumberInput, .stSelectbox { margin-bottom:15px; }
</style>
""", unsafe_allow_html=True)

# 加载数据和模型
@st.cache_resource
def load_resources():
    model = Model.load('gbm_model.pmml')
    dev = pd.read_csv('dev_finally.csv')
    vad = pd.read_csv('vad_finally.csv')
    return model, dev, vad

pmml_model, dev, vad = load_resources()

# 特征配置
feature_names = ['smoker', 'sex','carace', 'drink','sleep','Hypertension', 'Dyslipidemia',
                 'HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI']

# 侧边栏说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. 填写所有健康相关信息
    2. 点击预测按钮获取风险评估
    3. 查看详细解释和健康建议
    """)
    st.markdown("---")
    st.markdown("**特征说明**")
    st.caption("HHR Ratio: 家庭收入中位数与贫困线的比率")
    st.caption("INDFMPIR: 家庭贫困收入比")

# 主界面
st.title("❤️ 心脑血管共病风险预测系统")
st.markdown("---")

# 输入面板
with st.expander("📝 填写健康信息", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 基本信息")
        RIDAGEYR = st.slider("年龄 (岁)", 20, 80, 50, 
                           help="请选择您的实际年龄")
        sex = st.radio("性别", options=[1, 2],
                      format_func=lambda x: "♀️ 女性" if x == 1 else "♂️ 男性",
                      horizontal=True)
        carace = st.selectbox("种族/民族", options=[1, 2, 3, 4, 5], 
                            format_func=lambda x: ["墨西哥裔美国人", "其他西班牙裔", 
                                                 "非西班牙裔白人", "非西班牙裔黑人", 
                                                 "其他种族"][x-1])
        
    with col2:
        st.markdown("### 健康指标")
        BMXBMI = st.slider("身体质量指数 (kg/m²)", 11.5, 67.3, 25.0, 0.1,
                          help="正常范围: 18.5-24.9")
        HHR = st.slider("家庭收入比", 0.23, 1.67, 1.0, 0.01,
                       help="家庭收入中位数与贫困线的比率")
        INDFMPIR = st.slider("贫困收入比", 0.0, 5.0, 2.0, 0.1,
                           help="0表示最贫困，5表示最富裕")

# 生活方式部分
with st.expander("🏃 生活方式信息", expanded=True):
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 生活习惯")
        smoker = st.select_slider("吸烟状态", options=[1, 2, 3],
                             format_func=lambda x: ["从不吸烟", "曾经吸烟", 
                                                  "当前吸烟"][x-1])
        drink = st.radio("饮酒习惯", options=[1, 2],
                       format_func=lambda x: "否" if x == 1 else "是",
                       horizontal=True)
        
    with col4:
        st.markdown("### 健康史")
        Hypertension = st.toggle("高血压病史", value=False,
                               help="曾被诊断为高血压")
        Dyslipidemia = st.toggle("血脂异常", value=False,
                               help="胆固醇或甘油三酯异常")
        sleep = st.toggle("睡眠问题", value=False,
                        help="存在长期睡眠障碍")

# 实验室指标
with st.expander("🔬 实验室指标"):
    LBXWBCSI = st.number_input("白细胞计数 (10^9/L)", 1.4, 117.2, 6.0, 0.1,
                              help="正常范围: 4-10 ×10^9/L")
    LBXRBCSI = st.number_input("红细胞计数 (10^12/L)", 2.52, 7.9, 4.5, 0.1,
                              help="正常范围: 4.2-5.9 ×10^12/L")

# 预测和结果展示
if st.button("🚀 开始预测", use_container_width=True):
    # 数据预处理
    feature_values = [
        2 if smoker == "曾经吸烟" else 3 if smoker == "当前吸烟" else 1,
        2 if sex == "♂️ 男性" else 1,
        carace,
        2 if drink == "是" else 1,
        1 if sleep else 2,
        2 if Hypertension else 1,
        2 if Dyslipidemia else 1,
        HHR,
        RIDAGEYR,
        INDFMPIR,
        BMXBMI,
        LBXWBCSI,
        LBXRBCSI
    ]
    
    input_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # 执行预测
    try:
        prediction = pmml_model.predict(input_df)
        prob_0 = prediction['probability(1)'][0]
        prob_1 = prediction['probability(0)'][0]
        predicted_class = 1 if prob_1 > 0.436018256400085 else 0
        
        # 结果展示
        result_class = "high-risk" if predicted_class == 1 else "low-risk"
        result_text = """
        <div class='result-box {0}'>
            <h3>预测结果：{1}</h3>
            <div class='big-font'>共病概率：{2:.1f}%</div>
            <div class='big-font'>健康概率：{3:.1f}%</div>
            <progress value="{2}" max="100" style="width:100%; height:20px"></progress>
        </div>
        """.format(
            result_class,
            "⚠️ 高风险（建议立即咨询医生）" if predicted_class == 1 else "✅ 低风险（保持健康习惯）",
            prob_1*100,
            prob_0*100
        )
        st.markdown(result_text, unsafe_allow_html=True)
        
        # 健康建议
        advice = """
        ## 📌 健康建议
        {}
        """.format("""
        **高风险建议：**
        1. 立即预约心血管专科医生
        2. 控制血压和血脂水平
        3. 戒烟并限制酒精摄入
        4. 保持规律运动和健康饮食
        """ if predicted_class == 1 else """
        **低风险建议：**
        1. 保持当前健康生活方式
        2. 定期进行健康体检
        3. 维持正常体重范围
        4. 管理压力保证充足睡眠
        """)
        st.markdown(advice)
        
        # SHAP解释
        st.markdown("## 📊 特征影响分析（SHAP）")
        background = vad[feature_names].iloc[:100]
        
        def pmml_predict(data):
            return np.column_stack((
                pmml_model.predict(pd.DataFrame(data, columns=feature_names))['probability(0)'],
                pmml_model.predict(pd.DataFrame(data, columns=feature_names))['probability(1)']
            ))
        
        explainer = shap.KernelExplainer(pmml_predict, background)
        shap_values = explainer.shap_values(input_df)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[predicted_class], input_df, plot_type="bar", show=False)
        st.pyplot(fig)
        
        # LIME解释
        st.markdown("## 🔍 局部可解释分析（LIME）")
        lime_explainer = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['健康', '高风险'],
            verbose=False
        )
        lime_exp = lime_explainer.explain_instance(
            input_df.values.flatten(),
            pmml_predict
        )
        st.components.v1.html(lime_exp.as_html(show_table=True), height=600)
        
    except Exception as e:
        st.error(f"预测过程中发生错误：{str(e)}")

# 页脚
st.markdown("---")
st.markdown("**免责声明**：本工具仅用于科研参考，不能替代专业医疗建议。")
