# # src/gender_classification/app.py - 100% MATCHES TRAINING DATA
# import streamlit as st
# import os
# import pandas as pd
# import numpy as np
# import pickle
# import pathlib

# # Correct relative path from src/gender_classification/app.py to root/artifacts/gender_classification/
# BASE_DIR = pathlib.Path(__file__).parent.parent.parent  # src/gender_classification/ -> src/ -> root/
# ARTIFACTS_PATH = BASE_DIR / "artifacts" / "gender_classification"

# st.set_page_config(page_title="Gender Predictor", layout="wide")
# st.title("👤 Gender Classifier")
# st.info("🏆 LogisticRegression F1=0.424 | Live Demo")

# # @st.cache_resource
# # def load_model():
# #     model = pickle.load(open(ARTIFACTS_PATH + "model.pkl", "rb"))
# #     preprocessor = pickle.load(open(ARTIFACTS_PATH + "preprocessor.pkl", "rb"))
# #     le = pickle.load(open(ARTIFACTS_PATH + "label_encoder.pkl", "rb"))
# #     return model, preprocessor, le

# @st.cache_resource
# def load_model():
#     model_path = ARTIFACTS_PATH / "model.pkl"
#     preprocessor_path = ARTIFACTS_PATH / "preprocessor.pkl"
#     le_path = ARTIFACTS_PATH / "label_encoder.pkl"
    
#     # Debug: Show paths (remove after fixing)
#     st.info(f"Loading from: {ARTIFACTS_PATH.resolve()}")
    
#     model = pickle.load(open(model_path, "rb"))
#     preprocessor = pickle.load(open(preprocessor_path, "rb"))
#     le = pickle.load(open(le_path, "rb"))
#     return model, preprocessor, le

# model, preprocessor, le = load_model()
# st.success("✅ Model Ready!")

# # PERFECT INPUT MATCHING TRAINING
# st.header("🔮 Predict")
# col1, col2, col3 = st.columns(3)
# name = col1.text_input("👤 Name", "John Smith")
# age = int(col2.slider("🎂 Age", 18, 65, 28))
# company = col3.text_input("🏢 Company", "google")

# if st.button("🚀 PREDICT", type="primary"):
#     try:
#         # 🔥 EXACT SAME COLUMNS + ORDER AS TRAINING DATA
#         input_df = pd.DataFrame({
#             'code': ['CODE_001'],           # ← 1st column (required!)
#             'company': [company.lower()],   # ← 2nd (lowercased)
#             'name': [name],                 # ← 3rd  
#             'gender': ['unknown'],          # ← 4th (dummy for cleaning)
#             'age': [age]                    # ← 5th
#         })
        
#         st.write("✅ Input matches training:", input_df.columns.tolist())
        
#         # Transform (drop gender before preprocessing)
#         X = preprocessor.transform(input_df.drop(columns=['gender']))
        
#         # Predict
#         pred = model.predict(X)[0]
#         probs = model.predict_proba(X)[0]
#         gender = le.inverse_transform([int(pred)])[0]
        
#         # Results
#         st.balloons()
#         st.markdown(f"### **{gender}** 🎯")
#         st.metric("Confidence", f"{np.max(probs):.1%}")
        
#         col1, col2 = st.columns(2)
#         col1.metric("Female", f"{probs[0]:.0f}%")
#         col2.metric("Male", f"{probs[1]:.0f}%")
        
#         if len(probs) > 2:
#             st.metric("None", f"{probs[2]:.0f}%")
            
#     except Exception as e:
#         st.error(f"❌ {str(e)}")
#         st.code(f"Debug: {input_df}")

# st.markdown("---")
# st.caption("MLOps Demo | F1=0.424")

###########################################################

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
ARTIFACTS_PATH = BASE_DIR / "artifacts" / "gender_classification"

st.set_page_config(page_title="Gender Predictor", layout="wide")
st.title("👤 Gender Classifier")
st.info("🏆 LogisticRegression F1=0.424")

@st.cache_resource
def load_model():
    model = pickle.load(open(ARTIFACTS_PATH / "model.pkl", "rb"))
    preprocessor = pickle.load(open(ARTIFACTS_PATH / "preprocessor.pkl", "rb"))
    le = pickle.load(open(ARTIFACTS_PATH / "label_encoder.pkl", "rb"))
    st.success("✅ Loaded!")
    return model, preprocessor, le

model, preprocessor, le = load_model()

st.header("🔮 Predict")
col1, col2, col3 = st.columns(3)
name = col1.text_input("👤 Name", "John Smith")
age = col2.slider("🎂 Age", 18, 65, 28)
company = col3.text_input("🏢 Company", "google")

if st.button("🚀 PREDICT", type="primary"):
    try:
        # STEP 1: Raw input (like CSV)
        raw_df = pd.DataFrame({
            'code': ['CODE_001'],
            'company': [company],
            'name': [name],
            'gender': ['unknown'],  # Dummy
            'age': [age]
        })
        
        # STEP 2: EXACT get_data_cleaner() replica
        raw_df['age'] = pd.to_numeric(raw_df['age'], errors='coerce').fillna(30)
        raw_df['company'] = raw_df['company'].astype(str).str.strip().str.lower()
        raw_df['name'] = raw_df['name'].astype(str).str.strip()
        raw_df['gender'] = raw_df['gender'].astype(str).str.strip().str.lower()
        
        # CRITICAL: Add age_group with EXACT bins
        age_bins = [18, 25, 35, 50, 120]
        age_labels = ['18-25', '26-35', '36-50', '50+']
        raw_df['age_group'] = pd.cut(raw_df['age'], bins=age_bins, labels=age_labels,
                                   right=True, include_lowest=True)
        
        # STEP 3: Select EXACT feature columns
        feature_cols = ['name', 'age', 'age_group', 'company']
        X = raw_df[feature_cols]
        
        # st.success(f"✅ Ready for transform: {X.shape} | cols: {list(X.columns)}")
        st.dataframe(X)
        
        # STEP 4: Transform → Predict
        X_transformed = preprocessor.transform(X)
        pred = model.predict(X_transformed)[0]
        probs = model.predict_proba(X_transformed)[0]
        gender = le.inverse_transform([int(pred)])[0]
        
        st.balloons()
        st.markdown(f"### **{gender}** 🎯")
        st.metric("Confidence", f"{max(probs):.1%}")
        st.write(f"Female: {probs[0]:.0f}% | Male: {probs[1]:.0f}%")
        
    except Exception as e:
        st.error(f"❌ {str(e)}")

st.caption("Fixed: Exact replica of data_transformation.py")

