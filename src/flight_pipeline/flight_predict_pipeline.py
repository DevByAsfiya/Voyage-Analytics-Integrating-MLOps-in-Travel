import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","flight_price","model.pkl")
            preprocessor_path=os.path.join('artifacts','flight_price','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( 
                self,
                from_: str,
                to: str,
                flightType: str,
                time: float,
                distance: float,
                agency: str,
                date: str           # Pass as "MM/DD/YYYY"
    ):

        self.from_ = from_
        self.to = to
        self.flightType = flightType
        self.time = time
        self.distance = distance
        self.agency = agency
        self.date = date



    def get_data_as_data_frame(self):
        try:
            dt = pd.to_datetime(self.date)  # or use "%Y-%m-%d" as per your UI
            custom_data_input_dict = {
                    "from": [self.from_],
                    "to": [self.to],
                    "flightType": [self.flightType],
                    "time": [self.time],
                    "distance": [self.distance],
                    "agency": [self.agency],
                    "year": [dt.year],
                    "month": [dt.month],
                    "day": [dt.day],
                    "dayofweek": [dt.dayofweek],
                    "week_no": [dt.isocalendar().week]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)