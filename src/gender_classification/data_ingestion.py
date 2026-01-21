import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.gender_classification.data_transformation import DataTransformation
from src.gender_classification.data_transformation import DataTransformationConfig

from src.gender_classification.model_trainer import ModelTrainerConfig
from src.gender_classification.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'gender_classification', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'gender_classification', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'gender_classification', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('data/users.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()
#     train_data,test_data=obj.initiate_data_ingestion()

#     data_transformation=DataTransformation()
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


if __name__=="__main__":
    logging.info("🚀 COMPLETE GENDER CLASSIFICATION PIPELINE")
    
    # 1. Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # 2. Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_data, test_data)
    
    # 3. Model Training - SIMPLIFIED!
    trainer = ModelTrainer()
    best_f1 = trainer.initiate_model_trainer(
        transformation.transformation_config.transformed_train_path,
        transformation.transformation_config.transformed_test_path,
        transformation.transformation_config.label_encoder_path  # Only this!
    )
    
    print(f"\n🎉 PIPELINE 100% COMPLETE!")
    print(f"🏆 Best Model F1: {best_f1:.3f}")
    print(f"📁 Production Ready: artifacts/gender_classification/model.pkl")


