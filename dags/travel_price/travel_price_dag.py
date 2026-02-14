from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# KEEP THIS AT THE TOP
sys.path.append('/opt/airflow')

# --- REMOVE HEAVY IMPORTS FROM HERE ---
# from src.flight_price.data_ingestion import DataIngestion  <-- DELETE THIS
# from src.flight_price.data_transformation import DataTransformation <-- DELETE THIS
# from src.flight_price.model_trainer import ModelTrainer <-- DELETE THIS

def run_ingestion(**kwargs):
    # --- MOVE IMPORT HERE ---
    from src.flight_price.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    kwargs['ti'].xcom_push(key='train_path', value=train_path)
    kwargs['ti'].xcom_push(key='test_path', value=test_path)

def run_transformation(**kwargs):
    # --- MOVE IMPORT HERE ---
    from src.flight_price.data_transformation import DataTransformation
    
    ti = kwargs['ti']
    train_path = ti.xcom_pull(key='train_path', task_ids='ingest_data')
    test_path = ti.xcom_pull(key='test_path', task_ids='ingest_data')
    
    transformation = DataTransformation()
    transformation.initiate_data_transformation(train_path, test_path)

def run_training(**kwargs):
    # --- MOVE IMPORT HERE ---
    from src.flight_price.model_trainer import ModelTrainer
    
    trainer = ModelTrainer()
    print(trainer.initiate_model_trainer())

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 0
}

with DAG('travel_price_prediction', 
         default_args=default_args, 
         schedule_interval='@weekly', 
         catchup=False) as dag:

    ingest = PythonOperator(task_id='ingest_data', python_callable=run_ingestion)
    transform = PythonOperator(task_id='transform_data', python_callable=run_transformation)
    train = PythonOperator(task_id='train_model', python_callable=run_training)

    ingest >> transform >> train
