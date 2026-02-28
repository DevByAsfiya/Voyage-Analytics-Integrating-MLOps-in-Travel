import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object


@dataclass
class ContentRecommenderConfig:
    # === BASE PATH (for NEW outputs only) ===
    recommender_data_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender")
    
    # === INPUTS (match DataTransformation locations) ===
    input_hotel_features_path: str = os.path.join("artifacts", "hotel_recommendation", "hotel_features.csv")
    input_user_profile_path: str = os.path.join("artifacts", "hotel_recommendation", "user_profile.csv")
    input_preprocessor_path: str = os.path.join("artifacts", "hotel_recommendation", "preprocessor.pkl")
    input_trips_all_path: str = os.path.join("artifacts", "hotel_recommendation", "trips_all.csv")
    
    # === OUTPUTS (NEW files in Content_Recommender/) ===
    output_preprocessor_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender", "content_preprocessor.pkl")
    output_hotel_matrix_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender", "hotel_matrix.npy")
    output_hotel_features_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender", "hotel_features.csv")
    output_recommender_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender", "content_recommender.pkl")
    output_recommender_info_path: str = os.path.join("artifacts", "hotel_recommendation", "Content_Recommender", "recommender_info.json")


class ContentRecommender:
    def __init__(self):
        self.config = ContentRecommenderConfig()
        self.preprocessor = None
        self.hotel_matrix = None
        self.hotel_features = None
        logging.info("ContentRecommender initialized")

    def initiate_content_recommender(self) -> str:
        try:
            logging.info("=== Content-Based Recommender Setup Started ===")
            
            # Create Content_Recommender folder
            os.makedirs(self.config.recommender_data_path, exist_ok=True)
            
            # Load input data
            hotel_features_input = pd.read_csv(self.config.input_hotel_features_path)
            user_profile = pd.read_csv(self.config.input_user_profile_path)
            
            logging.info("Loaded - hotels: %s, users: %s", 
                        hotel_features_input.shape, user_profile.shape)
            
            # === STEP 1: Create price_bucket (EXACTLY like your Colab) ===
            hotel_features_input["price_bucket"] = hotel_features_input["avg_price"].apply(
                lambda x: "low" if x < 150 else "medium" if x < 300 else "high"
            )
            
            # === STEP 2: Define Content-Based feature columns (EXACTLY like Colab) ===
            cat_cols = ["city", "most_common_season", "price_bucket"]
            num_cols = ["avg_price", "avg_days", "bookings"]  # Use 'popularity' as 'bookings'
            
            # Rename popularity to bookings for consistency with Colab
            # hotel_features_input = hotel_features_input.rename(columns={'popularity': 'bookings'})
            
            logging.info("Content features - cat: %s, num: %s", cat_cols, num_cols)
            
            # === STEP 3: Build & Fit Content-Specific Preprocessor (like Colab) ===
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ("num", StandardScaler(), num_cols),  # Scale numerics like Colab
                ]
            )
            
            # Transform hotel features to matrix (EXACTLY like Colab)
            feature_df = hotel_features_input[cat_cols + num_cols]
            self.hotel_matrix = self.preprocessor.fit_transform(feature_df)
            self.hotel_features = hotel_features_input.copy()
            
            logging.info("Hotel matrix shape: %s (n_hotels=%d, n_features=%d)", 
                        self.hotel_matrix.shape, len(self.hotel_features), self.hotel_matrix.shape[1])
            
            # === STEP 4: Test the recommender ===
            self.test_recommendation(user_profile)
            
            # === STEP 5: Save ALL components (EXACTLY like your Colab SAVE block) ===
            joblib.dump(self.preprocessor, self.config.output_preprocessor_path)
            np.save(self.config.output_hotel_matrix_path, self.hotel_matrix)
            self.hotel_features.to_csv(self.config.output_hotel_features_path, index=False)
            
            # Save COMPLETE recommender (like your Colab recommender_data dict)
            recommender_data = {
                'preprocessor': self.preprocessor,
                'hotel_matrix': self.hotel_matrix,
                'hotel_features': self.hotel_features,
                'recommend_new_user': self.recommend_new_user,
                'build_intent_vector': self.build_intent_vector
            }
            joblib.dump(recommender_data, self.config.output_recommender_path)
            
            # Save info
            info = {
                "n_hotels": len(self.hotel_features),
                "n_features": self.hotel_matrix.shape[1],
                "cat_cols": cat_cols,
                "num_cols": num_cols,
                "price_thresholds": {"low": 150, "medium": 300}
            }
            pd.DataFrame([info]).to_json(self.config.output_recommender_info_path, orient="records")
            
            logging.info("✅ SAVED COMPLETE SYSTEM:")
            logging.info("📁 %s/", self.config.recommender_data_path)
            logging.info("├── content_preprocessor.pkl")
            logging.info("├── hotel_matrix.npy")
            logging.info("├── hotel_features.csv")
            logging.info("└── content_recommender.pkl ← ALL-IN-ONE")
            
            # Print file sizes
            for file in os.listdir(self.config.recommender_data_path):
                size = os.path.getsize(os.path.join(self.config.recommender_data_path, file)) / 1024
                print(f"  {file} ({size:.1f} KB)")
            
            logging.info("=== Content Recommender Setup Completed ===")
            return self.config.output_recommender_path
            
        except Exception as e:
            logging.error("Error in initiate_content_recommender: %s", e)
            raise CustomException(e, sys)

    def build_intent_vector(self, city, season, price, days, price_bucket):
        """EXACTLY like your Colab - builds user intent vector"""
        row = pd.DataFrame([{
            "city": city,
            "most_common_season": season,
            "price_bucket": price_bucket,
            "avg_price": price,
            "avg_days": days,
            "bookings": self.hotel_features["bookings"].mean()  # Global average
        }])
        return self.preprocessor.transform(row)

    def recommend_new_user(self, city, season, price, days, top_n=5):
        """EXACTLY like your Colab - core recommendation logic"""
        # Determine price bucket
        price_bucket = "low" if price < 150 else "medium" if price < 300 else "high"
        
        # Build intent vector
        intent_vec = self.build_intent_vector(city, season, price, days, price_bucket)
        
        # Compute similarity for ALL hotels
        sims = cosine_similarity(intent_vec, self.hotel_matrix).flatten()
        self.hotel_features["score"] = sims
        
        # FILTER by exact city match FIRST (your key insight)
        city_hotels = self.hotel_features[self.hotel_features["city"] == city].copy()
        
        if len(city_hotels) > 0:
            result = city_hotels.sort_values("score", ascending=False).head(top_n)
        else:
            # Fallback
            result = self.hotel_features.sort_values("score", ascending=False).head(top_n)
            print(f"⚠️ No hotel found in {city}. Showing similar hotels.")
        
        return result[["hotel_name", "city", "avg_price", "avg_days", 
                      "most_common_season", "bookings", "score"]]

    def test_recommendation(self, user_profile):
        """Test with sample user - like your Colab demo"""
        try:
            example_user = user_profile.sample(1).iloc[0]
            city = example_user.get('fav_city', 'Mumbai')
            days = example_user.get('mean_days', 3)
            
            print(f"\n🧪 Testing for user (prefers {city}):")
            recs = self.recommend_new_user(
                city=city, 
                season="summer", 
                price=250, 
                days=days, 
                top_n=3
            )
            print(recs.round(3))
            logging.info("✅ Recommendation test passed")
        except Exception as e:
            logging.error("Recommendation test failed: %s", e)





###########################################################################################

# import os
# import sys
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from sklearn.model_selection import train_test_split
# import joblib

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, load_object

# @dataclass
# class ModelTrainerConfig:
#     # === BASE PATH (for NEW outputs only) ===
#     model_data_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data")
    
#     # === INPUTS (match DataTransformation locations - NO Model_Data/) ===
#     input_hotel_features_path: str = os.path.join("artifacts", "hotel_recommendation", "hotel_features.csv")
#     input_user_profile_path: str = os.path.join("artifacts", "hotel_recommendation", "user_profile.csv")
#     input_preprocessor_path: str = os.path.join("artifacts", "hotel_recommendation", "preprocessor.pkl")
#     input_trips_all_path: str = os.path.join("artifacts", "hotel_recommendation", "trips_all.csv")
    
#     # === OUTPUTS (NEW files in Model_Data/) ===
#     output_model_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "model.pkl")
#     output_metrics_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "model_metrics.json")
#     output_trips_train_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "trips_train.csv")
#     output_trips_test_path: str = os.path.join("artifacts", "hotel_recommendation", "Model_Data", "trips_test.csv")

# class ModelTrainer:
#     def __init__(self):
#         self.config = ModelTrainerConfig()
#         self.model = None
#         self.preprocessor = None
#         logging.info("ModelTrainer initialized")

#     def initiate_model_trainer(self) -> str:
#         try:
#             logging.info("=== Model training started ===")
            
#             # Create Model_Data folder for outputs
#             os.makedirs(self.config.model_data_path, exist_ok=True)
            
#             # Load existing artifacts (from DataTransformation)
#             self.preprocessor = load_object(self.config.input_preprocessor_path)
#             hotel_features = pd.read_csv(self.config.input_hotel_features_path)
#             user_profile = pd.read_csv(self.config.input_user_profile_path)
#             trips_all = pd.read_csv(self.config.input_trips_all_path)
            
#             logging.info("Loaded artifacts - trips_all: %s, hotels: %s, users: %s", 
#                         trips_all.shape, hotel_features.shape, user_profile.shape)
            
#             # Define feature columns (exact order matches preprocessor)
#             cat_cols = ['city', 'season', 'fav_city', 'fav_state', 'most_common_season']
#             num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']
#             feature_cols = cat_cols + num_cols
            
#             # Prepare training data
#             X = trips_all[feature_cols]
#             y = trips_all['label']
            
#             logging.info("Raw training data - X: %s, y: %s (label distribution: %.1f%% positive)", 
#                         X.shape, y.shape, 100*y.mean())
            
#             # Train/test split (reproducible)
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42, stratify=y
#             )
            
#             # Transform with existing preprocessor
#             X_train_transformed = self.preprocessor.transform(X_train)
#             X_test_transformed = self.preprocessor.transform(X_test)
            
#             logging.info("Transformed shapes - X_train: %s, X_test: %s", 
#                         X_train_transformed.shape, X_test_transformed.shape)
            
#             # ✅ FIXED: Train model FIRST
#             self.model = LogisticRegression(
#                 max_iter=2000,              # Fix convergence warning
#                 class_weight='balanced',    # Fix class imbalance
#                 random_state=42
#             )
#             self.model.fit(X_train_transformed, y_train)
            
#             # ✅ FIXED: Evaluate SECOND (after fit)
#             y_pred = self.model.predict(X_test_transformed)
#             y_proba = self.model.predict_proba(X_test_transformed)[:, 1]
            
#             accuracy = accuracy_score(y_test, y_pred)
#             auc = roc_auc_score(y_test, y_proba)
            
#             logging.info("Model trained. Accuracy: %.3f, ROC-AUC: %.3f", accuracy, auc)
#             print("\n📊 Model Performance:")
#             print(f"Accuracy: {accuracy:.3f}")
#             print(f"ROC-AUC:  {auc:.3f}")
#             print(classification_report(y_test, y_pred))
            
#             # Save trained model
#             save_object(self.config.output_model_path, self.model)
            
#             # Save comprehensive metrics
#             metrics = {
#                 "accuracy": float(accuracy), 
#                 "roc_auc": float(auc),
#                 "train_size": int(X_train_transformed.shape[0]),
#                 "test_size": int(X_test_transformed.shape[0]),
#                 "n_features": int(X_train_transformed.shape[1]),
#                 "class_0_support": int(sum(y_test == 0)),
#                 "class_1_support": int(sum(y_test == 1))
#             }
#             pd.DataFrame([metrics]).to_json(self.config.output_metrics_path, orient="records")
            
#             # Save preprocessed train/test data
#             train_df = pd.DataFrame(X_train_transformed)
#             train_df['label'] = y_train.values
#             train_df.to_csv(self.config.output_trips_train_path, index=False)
            
#             test_df = pd.DataFrame(X_test_transformed)
#             test_df['label'] = y_test.values
#             test_df.to_csv(self.config.output_trips_test_path, index=False)
            
#             logging.info("✅ Preprocessed data saved - train: %s, test: %s", 
#                         train_df.shape, test_df.shape)
            
#             # Test recommendation function
#             self.test_recommendation(hotel_features, user_profile)
            
#             logging.info("=== Model training completed successfully ===")
#             logging.info("Model saved to: %s", self.config.output_model_path)
#             return self.config.output_model_path
            
#         except Exception as e:
#             logging.error("Error in initiate_model_trainer: %s", e)
#             raise CustomException(e, sys)

#     def test_recommendation(self, hotel_features, user_profile):
#         """Test the recommendation scoring function"""
#         try:
#             example_user = user_profile.sample(1).iloc[0]
#             logging.info("Testing recommendations for user: %s (fav_city: %s)", 
#                         example_user.get('userCode', 'Unknown'), example_user.get('fav_city', 'Unknown'))
            
#             recs = self.score_user_hotels(example_user, hotel_features, top_n=5)
#             print("\n🏨 Sample recommendations:")
#             print(recs[['hotel_id', 'hotel_name', 'city', 'avg_price', 'score']].round(3))
#             logging.info("Recommendation test completed - top hotel: %s (score: %.3f)", 
#                         recs.iloc[0]['hotel_id'], recs.iloc[0]['score'])
#         except Exception as e:
#             logging.error("Recommendation test failed: %s", e)
#             print("Recommendation test skipped due to error")

#     def score_user_hotels(self, user_row, hotel_features, top_n=5):
#         """Core recommendation scoring function"""
#         df = hotel_features.copy()
        
#         # Fill user-specific features for all hotels
#         df['fav_city'] = user_row['fav_city']
#         # df['fav_state'] = user_row['fav_state']
#         df['total_per_day'] = df['avg_price']  # Proxy
#         df['days'] = user_row['mean_days']     # User preference
#         df['season'] = df['most_common_season']
        
#         # Define same feature columns as training
#         cat_cols = ['city', 'season', 'fav_city', 'fav_state', 'most_common_season']
#         num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']
#         feature_cols = cat_cols + num_cols
        
#         X = df[feature_cols]
        
#         # Transform with fitted preprocessor + predict
#         X_transformed = self.preprocessor.transform(X)
#         scores = self.model.predict_proba(X_transformed)[:, 1]  # Probability of label=1
#         df['score'] = scores
        
#         # Return top recommendations
#         return df.sort_values('score', ascending=False).head(top_n)

