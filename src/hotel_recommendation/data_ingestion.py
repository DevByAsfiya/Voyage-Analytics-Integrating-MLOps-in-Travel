import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split  # Keep for future, but unused now

from src.exception import CustomException
from src.logger import logging

# Updated imports for Content-Based pipeline
from src.hotel_recommendation.data_transformation import HotelDataTransformation
from src.hotel_recommendation.model_trainer import ContentRecommender  # Your new file


@dataclass
class DataIngestionConfig:
    # Simplified: ONLY raw data needed for content-based
    raw_data_path: str = os.path.join('artifacts', 'hotel_recommendation', 'raw_hotel_data.csv')
    raw_data_source: str = 'data/hotels.csv'  # Configurable input file


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("DataIngestion initialized for Content-Based Pipeline")

    def initiate_data_ingestion(self) -> str:
        """Simplified: Only saves raw data (no train/test split needed)"""
        logging.info("=== Data Ingestion Started ===")
        try:
            # Load raw data
            df = pd.read_csv(self.ingestion_config.raw_data_source)
            logging.info('Raw dataset loaded - shape: %s', df.shape)
            print(f"📊 Loaded {df.shape[0]:,} hotel bookings")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved: %s", self.ingestion_config.raw_data_path)
            
            logging.info("=== Data Ingestion Completed ===")
            return self.ingestion_config.raw_data_path  # Only raw path needed
            
        except Exception as e:
            logging.error("Error in data ingestion: %s", e)
            raise CustomException(e, sys)


# 🚀 SINGLE MAIN BLOCK - Runs COMPLETE Pipeline
if __name__ == "__main__":
    logging.info("🎬 COMPLETE CONTENT-BASED RECOMMENDATION PIPELINE")
    print("=" * 70)
    
    try:
        # === STEP 1: Data Ingestion ===
        print("📥 STEP 1: Data Ingestion...")
        ingestion = DataIngestion()
        raw_path = ingestion.initiate_data_ingestion()
        print(f"✅ Raw data: {raw_path}")
        
        # === STEP 2: Data Transformation ===
        print("\n🔄 STEP 2: Data Transformation...")
        transformer = HotelDataTransformation()
        hotel_features_path = transformer.initiate_data_transformation(raw_path)
        print(f"✅ Hotel features: {hotel_features_path}")
        
        # === STEP 3: Content Recommender Setup ===
        print("\n🤖 STEP 3: Content Recommender...")
        recommender = ContentRecommender()
        recommender_path = recommender.initiate_content_recommender()
        print(f"✅ Recommender: {recommender_path}")
        
        # === FINAL SUMMARY ===
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("📁 Artifacts created:")
        print("   artifacts/hotel_recommendation/")
        print("   ├── hotel_features.csv")
        print("   ├── user_profile.csv")
        print("   └── Content_Recommender/ (5 files)")
        print("🚀 Deploy: joblib.load('content_recommender.pkl')")
        print("="*70)
        
        logging.info("✅ COMPLETE PIPELINE SUCCESS")
        
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)



# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass

# # from src.hotel_recommendation.data_transformation import DataTransformation
# from src.hotel_recommendation.data_transformation import HotelDataTransformation

# from src.hotel_recommendation.model_trainer import ModelTrainerConfig
# from src.hotel_recommendation.model_trainer import ModelTrainer

# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join('artifacts', 'hotel_recommendation', 'train.csv')
#     test_data_path: str = os.path.join('artifacts', 'hotel_recommendation', 'test.csv')
#     raw_data_path: str = os.path.join('artifacts', 'hotel_recommendation', 'raw.csv')

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config=DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info("Entered the data ingestion method or component")
#         try:
#             df=pd.read_csv('data/hotels.csv')
#             logging.info('Read the dataset as dataframe')

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

#             logging.info("Train test split initiated")
#             train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

#             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

#             # logging.info("Ingestion of the data is completed")

#             # return(
#             #     self.ingestion_config.train_data_path,
#             #     self.ingestion_config.test_data_path

#             # )
#             logging.info("Ingestion of the data is completed")

#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path,
#                 self.ingestion_config.raw_data_path  # Add raw path for transformation
#             )

                    


#         except Exception as e:
#             raise CustomException(e,sys)
        

# if __name__ == "__main__":
#     ingestion = DataIngestion()
#     train_path, test_path, raw_path = ingestion.initiate_data_ingestion()  # Now returns 3 paths
    
#     transformer = HotelDataTransformation()
#     train_arr, test_arr, prep_path = transformer.initiate_data_transformation(raw_path)
    
#     print("Train array shape:", train_arr.shape)
#     print("Test array shape:", test_arr.shape)
#     print("Preprocessor saved at:", prep_path)

# #     obj=DataIngestion()
# #     obj.initiate_data_ingestion()
# #     train_data,test_data=obj.initiate_data_ingestion()

# #     data_transformation=DataTransformation()
# #     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

#     # modeltrainer=ModelTrainer()
#     # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

# if __name__ == "__main__":
#     logging.info("=== COMPLETE HOTEL RECOMMENDATION PIPELINE STARTED ===")
    
#     # Step 1: Data Ingestion
#     ingestion = DataIngestion()
#     train_path, test_path, raw_path = ingestion.initiate_data_ingestion()
#     logging.info("✅ Data ingestion completed - raw: %s", raw_path)
    
#     # Step 2: Data Transformation
#     transformer = HotelDataTransformation()
#     train_arr, test_arr, prep_path = transformer.initiate_data_transformation(raw_path)
#     logging.info("✅ Data transformation completed - prep: %s", prep_path)
#     print("Train array shape:", train_arr.shape)
#     print("Test array shape:", test_arr.shape)
    
#     # Step 3: Model Training (NEW)
#     trainer = ModelTrainer()
#     model_path = trainer.initiate_model_trainer()
#     logging.info("✅ Model training completed - model: %s", model_path)
    
#     print("\n" + "="*60)
#     print("🎉 COMPLETE PIPELINE SUCCESS!")
#     print(f"📁 All artifacts saved in: {trainer.config.model_data_path}")
#     print(f"✅ Model ready for deployment: {model_path}")
#     print("="*60)
