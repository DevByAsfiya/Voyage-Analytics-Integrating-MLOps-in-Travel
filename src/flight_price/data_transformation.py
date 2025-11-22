import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'flight_price', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create a transformer for numerical and categorical columns.
        No imputation since there are no missing values in the dataset.
        """
        try:
            # Columns to be processed
            categorical_columns = [ 'from', 'to', 'flightType', 'agency', 'year', 'month', 'day', 'dayofweek', 'week_no' ]
            numerical_columns = [ 'time', 'distance' ]

            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Load train and test data, add date features, drop unused columns,
        apply transformations, and return ready-to-train numpy arrays.
        """
        try:
            # Load data from CSVs
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Feature engineering for date
            for df in [train_df, test_df]:
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['dayofweek'] = df['date'].dt.dayofweek
                df['week_no'] = df['date'].dt.isocalendar().week
                # Drop original date and irrelevant columns
                df.drop(columns=['date', 'travelCode', 'userCode'], inplace=True, errors='ignore')

            logging.info("Date feature engineering completed, dropped unnecessary columns.")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'price'

            # Separate input features and target for train and test sets
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object.")

            # Convert sparse matrices to dense arrays
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            print("Feature shape:", input_feature_train_arr.shape)
            print("Target shape:", np.array(target_feature_train_df).reshape(-1, 1).shape)

            # Concatenate features and target for training and testing arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)]

            print("train_df shape:", train_df.shape)
            print("input_feature_train_df shape:", input_feature_train_df.shape)
            print("target_feature_train_df shape:", target_feature_train_df.shape)


            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
