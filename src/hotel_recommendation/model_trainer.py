import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    # === BASE PATH (for NEW outputs only) ===
    model_data_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data")
    
    # === INPUTS (match DataTransformation locations - NO Model_Data/) ===
    input_hotel_features_path: str = os.path.join("artifacts", "hotel_recommendation", "hotel_features.csv")
    input_user_profile_path: str = os.path.join("artifacts", "hotel_recommendation", "user_profile.csv")
    input_preprocessor_path: str = os.path.join("artifacts", "hotel_recommendation", "preprocessor.pkl")
    input_trips_all_path: str = os.path.join("artifacts", "hotel_recommendation", "trips_all.csv")
    
    # === OUTPUTS (NEW files in Model_Data/) ===
    output_model_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "model.pkl")
    output_metrics_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "model_metrics.json")
    output_trips_train_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "trips_train.csv")
    output_trips_test_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "trips_test.csv")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.model = None
        self.preprocessor = None
        logging.info("ModelTrainer initialized")

    def initiate_model_trainer(self) -> str:
        try:
            logging.info("=== Model training started ===")
            
            # Create Model_Data folder for outputs
            os.makedirs(self.config.model_data_path, exist_ok=True)
            
            # Load existing artifacts (from DataTransformation)
            self.preprocessor = load_object(self.config.input_preprocessor_path)
            hotel_features = pd.read_csv(self.config.input_hotel_features_path)
            user_profile = pd.read_csv(self.config.input_user_profile_path)
            trips_all = pd.read_csv(self.config.input_trips_all_path)
            
            logging.info("Loaded artifacts - trips_all: %s, hotels: %s, users: %s", 
                        trips_all.shape, hotel_features.shape, user_profile.shape)
            
            # Define feature columns (exact order matches preprocessor)
            cat_cols = ['city', 'state', 'season', 'fav_city', 'fav_state', 'most_common_season']
            num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']
            feature_cols = cat_cols + num_cols
            
            # Prepare training data
            X = trips_all[feature_cols]
            y = trips_all['label']
            
            logging.info("Raw training data - X: %s, y: %s (label distribution: %.1f%% positive)", 
                        X.shape, y.shape, 100*y.mean())
            
            # Train/test split (reproducible)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Transform with existing preprocessor
            X_train_transformed = self.preprocessor.transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)
            
            logging.info("Transformed shapes - X_train: %s, X_test: %s", 
                        X_train_transformed.shape, X_test_transformed.shape)
            
            # ✅ FIXED: Train model FIRST
            self.model = LogisticRegression(
                max_iter=2000,              # Fix convergence warning
                class_weight='balanced',    # Fix class imbalance
                random_state=42
            )
            self.model.fit(X_train_transformed, y_train)
            
            # ✅ FIXED: Evaluate SECOND (after fit)
            y_pred = self.model.predict(X_test_transformed)
            y_proba = self.model.predict_proba(X_test_transformed)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            logging.info("Model trained. Accuracy: %.3f, ROC-AUC: %.3f", accuracy, auc)
            print("\n📊 Model Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"ROC-AUC:  {auc:.3f}")
            print(classification_report(y_test, y_pred))
            
            # Save trained model
            save_object(self.config.output_model_path, self.model)
            
            # Save comprehensive metrics
            metrics = {
                "accuracy": float(accuracy), 
                "roc_auc": float(auc),
                "train_size": int(X_train_transformed.shape[0]),
                "test_size": int(X_test_transformed.shape[0]),
                "n_features": int(X_train_transformed.shape[1]),
                "class_0_support": int(sum(y_test == 0)),
                "class_1_support": int(sum(y_test == 1))
            }
            pd.DataFrame([metrics]).to_json(self.config.output_metrics_path, orient="records")
            
            # Save preprocessed train/test data
            train_df = pd.DataFrame(X_train_transformed)
            train_df['label'] = y_train.values
            train_df.to_csv(self.config.output_trips_train_path, index=False)
            
            test_df = pd.DataFrame(X_test_transformed)
            test_df['label'] = y_test.values
            test_df.to_csv(self.config.output_trips_test_path, index=False)
            
            logging.info("✅ Preprocessed data saved - train: %s, test: %s", 
                        train_df.shape, test_df.shape)
            
            # Test recommendation function
            self.test_recommendation(hotel_features, user_profile)
            
            logging.info("=== Model training completed successfully ===")
            logging.info("Model saved to: %s", self.config.output_model_path)
            return self.config.output_model_path
            
        except Exception as e:
            logging.error("Error in initiate_model_trainer: %s", e)
            raise CustomException(e, sys)

    def test_recommendation(self, hotel_features, user_profile):
        """Test the recommendation scoring function"""
        try:
            example_user = user_profile.sample(1).iloc[0]
            logging.info("Testing recommendations for user: %s (fav_city: %s)", 
                        example_user.get('userCode', 'Unknown'), example_user.get('fav_city', 'Unknown'))
            
            recs = self.score_user_hotels(example_user, hotel_features, top_n=5)
            print("\n🏨 Sample recommendations:")
            print(recs[['hotel_id', 'hotel_name', 'city', 'avg_price', 'score']].round(3))
            logging.info("Recommendation test completed - top hotel: %s (score: %.3f)", 
                        recs.iloc[0]['hotel_id'], recs.iloc[0]['score'])
        except Exception as e:
            logging.error("Recommendation test failed: %s", e)
            print("Recommendation test skipped due to error")

    def score_user_hotels(self, user_row, hotel_features, top_n=5):
        """Core recommendation scoring function"""
        df = hotel_features.copy()
        
        # Fill user-specific features for all hotels
        df['fav_city'] = user_row['fav_city']
        df['fav_state'] = user_row['fav_state']
        df['total_per_day'] = df['avg_price']  # Proxy
        df['days'] = user_row['mean_days']     # User preference
        df['season'] = df['most_common_season']
        
        # Define same feature columns as training
        cat_cols = ['city', 'state', 'season', 'fav_city', 'fav_state', 'most_common_season']
        num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']
        feature_cols = cat_cols + num_cols
        
        X = df[feature_cols]
        
        # Transform with fitted preprocessor + predict
        X_transformed = self.preprocessor.transform(X)
        scores = self.model.predict_proba(X_transformed)[:, 1]  # Probability of label=1
        df['score'] = scores
        
        # Return top recommendations
        return df.sort_values('score', ascending=False).head(top_n)

