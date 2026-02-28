import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Add StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class HotelDataTransformationConfig:
    # === Core outputs for Content Recommender ===
    hotel_features_path: str = os.path.join(
        "artifacts", "hotel_recommendation", "hotel_features.csv"
    )
    user_profile_path: str = os.path.join(
        "artifacts", "hotel_recommendation", "user_profile.csv"
    )
    # Remove unused: preprocessor.pkl, trips_all.csv, train/test arrays


class HotelDataTransformation:
    def __init__(self):
        self.config = HotelDataTransformationConfig()
        logging.info("HotelDataTransformation initialized for Content-Based Recommender")

    @staticmethod
    def month_to_season(m: int) -> str:
        if m in [12, 1, 2]: return "winter"
        if m in [3, 4, 5]: return "summer" 
        if m in [6, 7, 8]: return "autumn"
        return "spring"

    def _prepare_base_tables(self, df: pd.DataFrame):
        """Optimized: Only create hotel_features + user_profile needed for content rec"""
        try:
            logging.info("Starting optimized feature engineering. Shape: %s", df.shape)
            
            df = df.copy()
            
            # === BASIC CLEANING (unchanged) ===
            df["travelCode"] = df["travelCode"].astype(int)
            df["userCode"] = df["userCode"].astype(int)
            for col in ["name", "place"]:
                df[col] = df[col].astype("string").str.strip()
            
            df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
            df["days"] = df["days"].astype(int)
            df["price"] = df["price"].astype(float)
            df["total"] = df["total"].astype(float)
            
            # === CITY/STATE EXTRACTION (unchanged) ===
            city_state = df["place"].str.extract(r"^(.*)\s+\((.*)\)$")
            df["city"] = city_state[0].str.strip()
            df["state"] = city_state[1].str.strip()
            
            # === DATE FEATURES (unchanged) ===
            df["month"] = df["date"].dt.month
            df["season"] = df["month"].apply(self.month_to_season)
            
            # === HOTEL ID (unchanged) ===
            df["hotel_id"] = df["name"].str.cat(df["city"], sep=" | ")
            
            # === HOTEL FEATURES (optimized - add price_bucket for Colab) ===
            logging.info("Creating hotel_features...")
            hotel_features = (
                df.groupby("hotel_id")
                .agg(
                    hotel_name=("name", "first"),
                    city=("city", "first"),
                    state=("state", "first"),
                    avg_price=("price", "mean"),
                    min_price=("price", "min"),
                    max_price=("price", "max"),
                    avg_days=("days", "mean"),
                    popularity=("travelCode", "count"),  # Renamed to 'bookings' later
                    most_common_season=("season", lambda x: x.mode().iat[0]),
                )
                .reset_index()
            )
            hotel_features["avg_price"] = hotel_features["avg_price"].round(2)
            
            # ✅ ADDED: price_bucket for Colab compatibility
            hotel_features["price_bucket"] = hotel_features["avg_price"].apply(
                lambda x: "low" if x < 150 else "medium" if x < 300 else "high"
            )
            
            # Rename popularity → bookings (Colab naming)
            hotel_features = hotel_features.rename(columns={'popularity': 'bookings'})
            
            logging.info("hotel_features shape: %s (ready for content rec)", hotel_features.shape)
            
            # === USER PROFILE (keep for testing - optional for pure content rec) ===
            logging.info("Creating user_profile...")
            user_profile = (
                df.groupby("userCode")
                .agg(
                    trips_count=("travelCode", "count"),
                    mean_price=("price", "mean"),
                    mean_days=("days", "mean"),
                    fav_city=("city", lambda x: x.mode().iat[0]),
                    fav_state=("state", lambda x: x.mode().iat[0]),
                )
                .reset_index()
            )
            user_profile["mean_price"] = user_profile["mean_price"].round(2)
            user_profile["mean_days"] = user_profile["mean_days"].round(1)
            logging.info("user_profile shape: %s", user_profile.shape)
            
            return hotel_features, user_profile
            
        except Exception as e:
            logging.error("Error in _prepare_base_tables: %s", e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_path: str):
        """Simplified: Only saves hotel_features.csv + user_profile.csv"""
        try:
            logging.info("=== Content-Based Data Transformation Started ===")
            logging.info("Raw path: %s", raw_path)
            
            # Load & engineer
            df = pd.read_csv(raw_path)
            hotel_features, user_profile = self._prepare_base_tables(df)
            
            # Create directories & save (ONLY what content recommender needs)
            os.makedirs(os.path.dirname(self.config.hotel_features_path), exist_ok=True)
            hotel_features.to_csv(self.config.hotel_features_path, index=False)
            user_profile.to_csv(self.config.user_profile_path, index=False)
            
            logging.info("✅ Saved for Content Recommender:")
            logging.info("  - %s (%d hotels)", self.config.hotel_features_path, len(hotel_features))
            logging.info("  - %s (%d users)", self.config.user_profile_path, len(user_profile))
            
            # ✅ Print Colab-ready columns
            print("\n📊 Hotel features ready for content recommender:")
            content_cols = ["city", "most_common_season", "price_bucket", "avg_price", "avg_days", "bookings"]
            print(f"   Content columns: {content_cols}")
            print(f"   Sample: {hotel_features[content_cols].head(1).to_dict()}")
            
            logging.info("=== Content-Based Data Transformation Completed ===")
            return self.config.hotel_features_path  # Return main artifact path
            
        except Exception as e:
            logging.error("Error in initiate_data_transformation: %s", e)
            raise CustomException(e, sys)

##############################################################

# import os
# import sys
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object


# @dataclass
# class HotelDataTransformationConfig:
#     preprocessor_obj_file_path: str = os.path.join(
#         "artifacts", "hotel_recommendation", "preprocessor.pkl"
#     )
#     hotel_features_path: str = os.path.join(
#         "artifacts", "hotel_recommendation", "hotel_features.csv"
#     )
#     user_profile_path: str = os.path.join(
#         "artifacts", "hotel_recommendation", "user_profile.csv"
#     )
#     trips_path: str = os.path.join(
#         "artifacts", "hotel_recommendation", "trips_all.csv"
#     )


# class HotelDataTransformation:
#     def __init__(self):
#         self.config = HotelDataTransformationConfig()
#         logging.info("HotelDataTransformation initialized with config: %s",
#                      self.config)

#     @staticmethod
#     def month_to_season(m: int) -> str:
#         if m in [12, 1, 2]:
#             return "winter"
#         if m in [3, 4, 5]:
#             return "summer"
#         if m in [6, 7, 8]:
#             return "autumn"
#         return "spring"

#     def _prepare_base_tables(self, df: pd.DataFrame):
#         try:
#             logging.info("Starting _prepare_base_tables. Input shape: %s", df.shape)

#             df = df.copy()
#             df["travelCode"] = df["travelCode"].astype(int)
#             df["userCode"] = df["userCode"].astype(int)

#             for col in ["name", "place"]:
#                 df[col] = df[col].astype("string").str.strip()

#             df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
#             df["days"] = df["days"].astype(int)
#             df["price"] = df["price"].astype(float)
#             df["total"] = df["total"].astype(float)

#             city_state = df["place"].str.extract(r"^(.*)\s+\((.*)\)$")
#             df["city"] = city_state[0].str.strip()
#             df["state"] = city_state[1].str.strip()

#             df["year"] = df["date"].dt.year
#             df["month"] = df["date"].dt.month
#             df["day_of_week"] = df["date"].dt.dayofweek
#             df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)
#             df["season"] = df["month"].apply(self.month_to_season)

#             df["total_per_day"] = df["total"] / df["days"]

#             df["price_bucket"] = pd.qcut(
#                 df["price"], q=3, labels=["low", "medium", "high"]
#             )

#             df["stay_length_group"] = pd.cut(
#                 df["days"],
#                 bins=[0, 1, 2, 3, 4],
#                 labels=["1 night", "2 nights", "3 nights", "4 nights"],
#                 include_lowest=True,
#                 right=True,
#             )

#             df["hotel_id"] = df["name"].str.cat(df["city"], sep=" | ")

#             logging.info("Aggregating hotel_features.")
#             hotel_features = (
#                 df.groupby("hotel_id")
#                 .agg(
#                     hotel_name=("name", "first"),
#                     city=("city", "first"),
#                     state=("state", "first"),
#                     avg_price=("price", "mean"),
#                     min_price=("price", "min"),
#                     max_price=("price", "max"),
#                     avg_days=("days", "mean"),
#                     popularity=("travelCode", "count"),
#                     most_common_season=("season", lambda x: x.mode().iat[0]),
#                 )
#                 .reset_index()
#             )
#             hotel_features["avg_price"] = hotel_features["avg_price"].round(2)
#             logging.info("hotel_features shape: %s", hotel_features.shape)

#             logging.info("Aggregating user_profile.")
#             user_profile = (
#                 df.groupby("userCode")
#                 .agg(
#                     trips_count=("travelCode", "count"),
#                     mean_price=("price", "mean"),
#                     mean_days=("days", "mean"),
#                     fav_city=("city", lambda x: x.mode().iat[0]),
#                     fav_state=("state", lambda x: x.mode().iat[0]),
#                     fav_price_bucket=("price_bucket", lambda x: x.mode().iat[0]),
#                 )
#                 .reset_index()
#             )
#             user_profile["mean_price"] = user_profile["mean_price"].round(2)
#             user_profile["mean_days"] = user_profile["mean_days"].round(1)
#             logging.info("user_profile shape: %s", user_profile.shape)

#             logging.info("Building trips table and label.")
#             trips = df[
#                 ["userCode", "hotel_id", "total_per_day", "days",
#                  "city", "state", "season"]
#             ]

#             trips = trips.merge(
#                 user_profile[["userCode", "fav_city", "fav_state", "fav_price_bucket"]],
#                 on="userCode",
#                 how="left",
#             )

#             trips = trips.merge(
#                 hotel_features[
#                     ["hotel_id", "avg_price", "avg_days",
#                      "popularity", "most_common_season"]
#                 ],
#                 on="hotel_id",
#                 how="left",
#             )

#             trips["label"] = (trips["fav_city"] == trips["city"]).astype(int)
#             logging.info("trips shape: %s; positive labels: %d",
#                          trips.shape, trips["label"].sum())

#             return hotel_features, user_profile, trips

#         except Exception as e:
#             logging.error("Error in _prepare_base_tables: %s", e)
#             raise CustomException(e, sys)


#     def _get_preprocessor(self, cat_cols, num_cols, X_train):  # ← Add X_train parameter
#         try:
#             logging.info("Creating ColumnTransformer. Categorical: %s; Numeric: %s",
#                         cat_cols, num_cols)
            
#             # Filter valid columns using X_train
#             valid_cat_cols = [col for col in cat_cols if X_train[col].nunique() > 1]
#             valid_num_cols = [col for col in num_cols if X_train[col].notna().sum() > 0]
            
#             logging.info("Valid cat cols: %s", valid_cat_cols)
#             logging.info("Valid num cols: %s", valid_num_cols)
            
#             if not valid_cat_cols and not valid_num_cols:
#                 logging.warning("No valid columns found, using empty preprocessor")
#                 from sklearn.preprocessing import FunctionTransformer
#                 return FunctionTransformer(lambda X: X, validate=False)
            
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), valid_cat_cols),
#                     ("num", "passthrough", valid_num_cols),
#                 ],
#                 remainder="drop",
#             )
#             return preprocessor
#         except Exception as e:
#             logging.error("Error in _get_preprocessor: %s", e)
#             raise CustomException(e, sys)



#     def initiate_data_transformation(self, raw_path: str):
#         """
#         Read raw hotel bookings, build engineered tables,
#         split trips into train/test, build preprocessor,
#         and return matrices.
#         """
#         try:
#             logging.info("=== Hotel data transformation started ===")
#             logging.info("Raw path: %s", raw_path)

#             df = pd.read_csv(raw_path)
#             logging.info("Loaded raw df shape: %s", df.shape)

#             hotel_features, user_profile, trips = self._prepare_base_tables(df)

#             os.makedirs(os.path.dirname(self.config.hotel_features_path),
#                         exist_ok=True)
#             hotel_features.to_csv(self.config.hotel_features_path, index=False)
#             user_profile.to_csv(self.config.user_profile_path, index=False)
#             trips.to_csv(self.config.trips_path, index=False)
#             logging.info("Saved engineered tables to artifacts folder.")

#             cat_cols = [
#                 "city",
#                 # "state",
#                 "season",
#                 "fav_city",
#                 "fav_state",
#                 "most_common_season",
#             ]
#             num_cols = ["total_per_day", "days", "avg_price", "avg_days", "popularity"]

#             X = trips[cat_cols + num_cols]
#             y = trips["label"]

#             logging.info("Full X shape: %s, y shape: %s", X.shape, y.shape)

#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42, stratify=y
#             )
#             logging.info("X_train: %s, y_train: %s", X_train.shape, y_train.shape)
#             logging.info("X_test : %s, y_test : %s", X_test.shape, y_test.shape)

#             preprocessor = self._get_preprocessor(cat_cols, num_cols, X_train)

#             logging.info("Fitting preprocessor on X_train.")
#             X_train_arr = preprocessor.fit_transform(X_train)
#             X_test_arr = preprocessor.transform(X_test)
#             logging.info("X_train_arr shape: %s, X_test_arr shape: %s",
#                          X_train_arr.shape, X_test_arr.shape)

#             y_train_arr = y_train.to_numpy().reshape(-1, 1)
#             y_test_arr = y_test.to_numpy().reshape(-1, 1)

#             train_arr = np.c_[X_train_arr, y_train_arr]
#             test_arr = np.c_[X_test_arr, y_test_arr]
#             logging.info("Final train_arr shape: %s, test_arr shape: %s",
#                          train_arr.shape, test_arr.shape)

#             save_object(
#                 file_path=self.config.preprocessor_obj_file_path,
#                 obj=preprocessor,
#             )
#             logging.info("Preprocessor saved to %s",
#                          self.config.preprocessor_obj_file_path)

#             logging.info("=== Hotel data transformation completed successfully ===")
#             return train_arr, test_arr, self.config.preprocessor_obj_file_path

#         except Exception as e:
#             logging.error("Error in initiate_data_transformation: %s", e)
#             raise CustomException(e, sys)
