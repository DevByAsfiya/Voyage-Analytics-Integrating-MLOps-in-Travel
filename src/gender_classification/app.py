# src/gender_classification/app.py - 100% MATCHES TRAINING DATA
import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle

# YOUR EXACT PATH
ARTIFACTS_PATH = "artifacts/gender_classification/"

st.set_page_config(page_title="Gender Predictor", layout="wide")
st.title("👤 Gender Classifier")
st.info("🏆 LogisticRegression F1=0.424 | Live Demo")

@st.cache_resource
def load_model():
    model = pickle.load(open(ARTIFACTS_PATH + "model.pkl", "rb"))
    preprocessor = pickle.load(open(ARTIFACTS_PATH + "preprocessor.pkl", "rb"))
    le = pickle.load(open(ARTIFACTS_PATH + "label_encoder.pkl", "rb"))
    return model, preprocessor, le

model, preprocessor, le = load_model()
st.success("✅ Model Ready!")

# PERFECT INPUT MATCHING TRAINING
st.header("🔮 Predict")
col1, col2, col3 = st.columns(3)
name = col1.text_input("👤 Name", "John Smith")
age = int(col2.slider("🎂 Age", 18, 65, 28))
company = col3.text_input("🏢 Company", "google")

if st.button("🚀 PREDICT", type="primary"):
    try:
        # 🔥 EXACT SAME COLUMNS + ORDER AS TRAINING DATA
        input_df = pd.DataFrame({
            'code': ['CODE_001'],           # ← 1st column (required!)
            'company': [company.lower()],   # ← 2nd (lowercased)
            'name': [name],                 # ← 3rd  
            'gender': ['unknown'],          # ← 4th (dummy for cleaning)
            'age': [age]                    # ← 5th
        })
        
        st.write("✅ Input matches training:", input_df.columns.tolist())
        
        # Transform (drop gender before preprocessing)
        X = preprocessor.transform(input_df.drop(columns=['gender']))
        
        # Predict
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        gender = le.inverse_transform([int(pred)])[0]
        
        # Results
        st.balloons()
        st.markdown(f"### **{gender}** 🎯")
        st.metric("Confidence", f"{np.max(probs):.1%}")
        
        col1, col2 = st.columns(2)
        col1.metric("Female", f"{probs[0]:.0f}%")
        col2.metric("Male", f"{probs[1]:.0f}%")
        
        if len(probs) > 2:
            st.metric("None", f"{probs[2]:.0f}%")
            
    except Exception as e:
        st.error(f"❌ {str(e)}")
        st.code(f"Debug: {input_df}")

st.markdown("---")
st.caption("MLOps Demo | F1=0.424")
