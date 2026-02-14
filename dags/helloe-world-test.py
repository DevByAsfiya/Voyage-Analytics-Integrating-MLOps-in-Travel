from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def print_hello():
    print("🎉 Hello World from Fabric Airflow!")
    print("✅ Git sync working perfectly!")
    print("📅 Run date:", datetime.now())
    return "SUCCESS"


def print_goodbye():
    print("👋 DAG complete successfully!")
    return "COMPLETE"


default_args = {
    'owner': 'test',
    'start_date': datetime(2026, 1, 1),
    'retries': 0
}


with DAG(
    'hello_world_test',
    default_args=default_args,
    description='Test Fabric Git sync',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['test', 'fabric']
) as dag:

    hello = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello
    )
    
    goodbye = PythonOperator(
        task_id='goodbye_task',
        python_callable=print_goodbye
    )
    
    hello >> goodbye
