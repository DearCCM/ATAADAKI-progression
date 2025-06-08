import streamlit as st
import numpy as np
import joblib

# Load model
aki_model = joblib.load('aki_model.pkl')

# Mappings
hydragogue_mapping = {"without": 0, "20mg": 1, ">200mg": 2}
gender_mapping = {"female": 0, "male": 1}

def predict_aki_probability(features):
    aki_prob = aki_model.predict_proba(features)[0][1]  # 返回为 AKI 的概率
    return aki_prob

def main():
    st.title('Prediction of AKI Progression After ATAAD Surgery')
    selected_content = st.radio("", ("Model Introduction", "AKI Progression Prediction"))

    if selected_content == "Model Introduction":
        st.subheader("Model Introduction")
        st.write("This platform predicts the risk of AKI progression to stage 3 after type A aortic dissection surgery.")

    elif selected_content == "AKI Progression Prediction":
        st.subheader("Input Patient Clinical Features")
        
        # 输入
        ventilation_time = st.number_input("Ventilation time (h)", value=0.0, format="%.2f")
        hydragogue = st.selectbox("Diuretics", ["without", "20mg", ">200mg"])
        HCO3 = st.number_input("HCO3- (mmol/L)", value=0.0, format="%.2f")
        ALB = st.number_input("Albumin (g/L)", value=0.0, format="%.2f")
        SCR = st.number_input("Scr (μmol/L)", value=0.0, format="%.2f")
        intraoperative_hemorrhage = st.number_input("Intraoperative hemorrhage (ml)", value=0.0, format="%.2f")
        gender = st.selectbox("Gender", ["female", "male"])
        platelet_rich_plasma = st.number_input("Platelet-rich plasma (ml)", value=0.0, format="%.2f")
        CRP = st.number_input("C-reactive protein (mg/L)", value=0.0, format="%.2f")
        Myo_time = st.number_input("Myocardial vascularization time (min)", value=0.0, format="%.2f")

        # 映射
        hydragogue_val = hydragogue_mapping[hydragogue]
        gender_val = gender_mapping[gender]

        features = np.array([[ventilation_time, hydragogue_val, HCO3, ALB, SCR,
                              intraoperative_hemorrhage, gender_val, platelet_rich_plasma,
                              CRP, Myo_time]])

        if st.button("Predict AKI Progression Probability"):
            prob = predict_aki_probability(features)
            st.write(f"Predicted Probability of AKI Progression: **{prob:.2%}**")
            if prob > 0.5:
                st.warning("High Risk: Consider enhanced renal protection.")
            else:
                st.success("Low Risk: Standard care is likely appropriate.")

if __name__ == '__main__':
    main()
