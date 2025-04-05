# ...（前面的导入和加载代码保持不变）

if st.button("Predict"):
    with st.spinner('Analyzing your data...'):
        # 创建输入数据框
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # 预测部分保持不变
        prediction = pmml_model.predict(input_df)
        prob_0 = prediction['probability(1)'][0]
        prob_1 = prediction['probability(0)'][0]
        predicted_class = 1 if prob_1 > 0.436018256400085 else 0
        probability = prob_1 if predicted_class == 1 else prob_0

        # ====================== 美化后的结果展示 ======================
        st.subheader("📊 Prediction Results")
        
        # 使用columns创建并排布局
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            # 使用容器突出显示结果
            with st.container():
                st.markdown("### Risk Assessment")
                if predicted_class == 1:
                    st.error("🚨 High Risk of Comorbidity")
                    st.metric(label="Probability", 
                            value=f"{prob_1*100:.1f}%", 
                            delta="High Risk Threshold: 43.6%")
                else:
                    st.success("✅ Low Risk of Comorbidity")
                    st.metric(label="Probability", 
                            value=f"{prob_0*100:.1f}%", 
                            delta="Safe Threshold: 56.4%")
                
                # 添加进度条可视化
                st.markdown("### Risk Probability")
                risk_level = prob_1 if predicted_class == 1 else prob_0
                st.progress(int(risk_level * 100))

        with result_col2:
            # 美化建议展示
            st.markdown("### 🩺 Health Recommendations")
            advice_container = st.container()
            with advice_container:
                if predicted_class == 1:
                    st.markdown("""
                    <div style="padding: 15px; border-radius: 10px; background: #fff3f3;">
                    <h4 style="color: #d60000">Recommended Actions:</h4>
                    <ul>
                        <li>🩺 Schedule a cardiology consultation immediately</li>
                        <li>📅 Arrange for comprehensive cardiovascular screening</li>
                        <li>🚭 Maintain strict smoking cessation (if applicable)</li>
                        <li>📉 Monitor blood pressure and cholesterol levels weekly</li>
                        <li>🏥 Consider hospitalization for further evaluation</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="padding: 15px; border-radius: 10px; background: #f0fff4;">
                    <h4 style="color: #008000">Preventive Measures:</h4>
                    <ul>
                        <li>🏥 Schedule annual cardiovascular check-up</li>
                        <li>🥗 Maintain balanced diet low in saturated fats</li>
                        <li>🏃‍♂️ Engage in 150 mins/week moderate exercise</li>
                        <li>📊 Monitor BMI and waist circumference monthly</li>
                        <li>💤 Ensure 7-8 hours quality sleep nightly</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

        # ====================== 增强的可视化解释 ======================
        st.markdown("---")
        st.subheader("🔍 Model Interpretation")
        
        # SHAP解释部分美化
        with st.expander("SHAP Force Plot Analysis", expanded=True):
            st.markdown("""
            **Explanation:** This diagram shows how each feature contributes to the prediction.
            - 🔴 Red bars push the prediction towards high risk
            - 🟢 Green bars reduce the risk probability
            """)
            
            plt.figure(figsize=(10, 4))
            if predicted_class == 1:
                shap.force_plot(explainer.expected_value[1], 
                              shap_values[0,:,1],
                              input_df.iloc[0],
                              matplotlib=True,
                              show=False,
                              fig=plt.gcf())
            else:
                shap.force_plot(explainer.expected_value[0], 
                              shap_values[0,:,0],
                              input_df.iloc[0],
                              matplotlib=True,
                              show=False,
                              fig=plt.gcf())
            
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()

        # LIME解释部分美化
        with st.expander("LIME Feature Impact Breakdown", expanded=True):
            st.markdown("""
            **Feature Impact:**  
            This table shows local feature importance for this specific prediction
            """)
            
            lime_exp = lime_explainer.explain_instance(
                data_row=input_df.values.flatten(),
                predict_fn=pmml_predict
            )
            
            # 创建两列布局显示LIME结果
            lime_col1, lime_col2 = st.columns([1, 3])
            
            with lime_col1:
                st.markdown("### 📈 Impact Summary")
                html = lime_exp.as_html(show_table=False)
                st.components.v1.html(html, height=300)
                
            with lime_col2:
                st.markdown("### 📋 Detailed Feature Analysis")
                df = pd.DataFrame(lime_exp.as_list(), columns=["Feature", "Impact"])
                st.dataframe(df.style.bar(subset=["Impact"], 
                                        align="mid", 
                                        color=["#ff6961", "#77dd77"]),
                            height=300)

        # 添加免责声明
        st.markdown("---")
        st.caption("""
        ℹ️ **Disclaimer:** This prediction is based on machine learning models and should not replace professional medical advice. 
        Always consult qualified healthcare providers for medical decisions.
        """)
