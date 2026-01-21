# src/gender_classification/data_transformation.py
# 🔥 ULTIMATE DEBUG VERSION - Will show EXACT problem

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'gender_classification', 'preprocessor.pkl')
    label_encoder_path: str = os.path.join('artifacts', 'gender_classification', 'label_encoder.pkl')
    transformed_train_path: str = os.path.join('artifacts', 'gender_classification', 'transformed_train.npz')
    transformed_test_path: str = os.path.join('artifacts', 'gender_classification', 'transformed_test.npz')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_cleaner(self, df):
        """Clean data"""
        try:
            df_clean = df.drop(columns=['code'], errors='ignore').copy()
            df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce').fillna(30).astype('Int64')
            
            for col in ['company', 'name', 'gender']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype('string').str.strip()
            
            if 'company' in df_clean.columns:
                df_clean['company'] = df_clean['company'].str.lower()
            if 'gender' in df_clean.columns:
                df_clean['gender'] = df_clean['gender'].str.lower()
            
            age_bins = [18, 25, 35, 50, 120]
            age_labels = ['18-25', '26-35', '36-50', '50+']
            df_clean['age_group'] = pd.cut(
                df_clean['age'],
                bins=age_bins,
                labels=age_labels,
                right=True,
                include_lowest=True
            )
            return df_clean
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """🔥 FULL DEBUG VERSION"""
        try:
            logging.info("🔍=== DEBUGGING TRANSFORMATION ===")
            
            # 1. Load & Clean
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df_clean = self.get_data_cleaner(train_df)
            test_df_clean = self.get_data_cleaner(test_df)
            
            feature_cols = ['name', 'age', 'age_group', 'company']
            target_col = 'gender'
            
            X_train = train_df_clean[feature_cols]
            y_train = train_df_clean[target_col]
            X_test = test_df_clean[feature_cols]
            y_test = test_df_clean[target_col]
            
            logging.info(f"🔍 X_train shape: {X_train.shape}")
            logging.info(f"🔍 X_train dtypes:\n{X_train.dtypes}")
            logging.info(f"🔍 Sample X_train:\n{X_train.head()}")
            
            # Encode target FIRST
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            
            # 🔥 STEP-BY-STEP DEBUG TRANSFORMERS
            logging.info("🔍=== TESTING EACH TRANSFORMER ===")
            
            # Test 1: TF-IDF only
            tfidf = TfidfVectorizer(max_features=100, lowercase=False)
            X_name_tfidf = tfidf.fit_transform(X_train['name'])
            logging.info(f"✅ TF-IDF shape: {X_name_tfidf.shape}")
            
            # Test 2: OneHotEncoder only  
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_cat_ohe = ohe.fit_transform(X_train[['age_group', 'company']])
            logging.info(f"✅ OHE shape: {X_cat_ohe.shape}")
            
            # Test 3: StandardScaler only
            scaler = StandardScaler()
            X_age_scaled = scaler.fit_transform(X_train[['age']])
            logging.info(f"✅ Scaler shape: {X_age_scaled.shape}")
            
            # 🔥 FULL ColumnTransformer
            logging.info("🔍=== FULL COLUMN TRANSFORMER ===")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('name_tfidf', TfidfVectorizer(max_features=100, lowercase=False), 'name'),
                    ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['age_group', 'company']),
                    ('num_scale', StandardScaler(), ['age'])
                ],
                remainder='drop'
            )
            
            # 🔥 TRANSFORM FULL DATA
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info(f"🎉 FINAL X_train_transformed: {X_train_transformed.shape}")
            
            # 🔥 DEBUG CONCATENATION
            logging.info(f"🔍 BEFORE CONCAT:")
            logging.info(f"   X_train_transformed.shape: {X_train_transformed.shape}")
            logging.info(f"   y_train_enc.shape:        {y_train_enc.shape}")
            
            # 🔥 SAFE CONCAT - Use hstack instead of c_
            from scipy.sparse import hstack
            if hasattr(X_train_transformed, 'toarray'):
                X_train_dense = X_train_transformed.toarray()
            else:
                X_train_dense = X_train_transformed
                
            y_train_2d = y_train_enc.reshape(-1, 1)
            train_arr = np.hstack([X_train_dense, y_train_2d])
            
            logging.info(f"✅ FINAL train_arr: {train_arr.shape}")
            
            # Test set
            X_test_transformed = preprocessor.transform(X_test)
            if hasattr(X_test_transformed, 'toarray'):
                X_test_dense = X_test_transformed.toarray()
            else:
                X_test_dense = X_test_transformed
                
            y_test_2d = y_test_enc.reshape(-1, 1)
            test_arr = np.hstack([X_test_dense, y_test_2d])
            
            # Save
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(self.transformation_config.preprocessor_obj_file_path, preprocessor)
            save_object(self.transformation_config.label_encoder_path, le)
            np.savez(self.transformation_config.transformed_train_path, arr=train_arr)
            np.savez(self.transformation_config.transformed_test_path, arr=test_arr)
            
            logging.info("🎉✅ 100% SUCCESS!")
            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            logging.error(f"❌ FULL ERROR: {str(e)}")
            raise CustomException(e, sys)
