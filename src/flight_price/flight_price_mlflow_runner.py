import os
import sys
import itertools
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.flight_price.data_ingestion import DataIngestion
from src.flight_price.data_transformation import DataTransformation
from src.flight_price.model_trainer import ModelTrainer, evaluate_models

def get_data_pipeline():
    """Shared data pipeline → returns arrays"""
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )
    
    X_train, y_train, X_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1]
    )
    
    return X_train, y_train, X_test, y_test

def train_single_model(model_name, X_train, y_train, X_test, y_test, **hyperparams):
    """Train one model with hyperparams → return model + score"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    
    models = {
        "RandomForest": RandomForestRegressor(random_state=42, **hyperparams),
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor(random_state=42, **hyperparams)
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

def main():
    mlflow.set_experiment("Flight_Price_Prediction_MultiModel")
    
    # Get shared data
    print("📥 Preparing data pipeline...")
    X_train, y_train, X_test, y_test = get_data_pipeline()
    
    # HYPERPARAMETER SWEEP + ALL MODELS
    rf_params = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    xgb_params = {"n_estimators": [50, 100], "learning_rate": [0.1, 0.2]}
    
    print("🔍 Multi-model hyperparameter sweep...")
    
    # Random Forest sweep
    for n_est, max_d in itertools.product(rf_params["n_estimators"], rf_params["max_depth"]):
        with mlflow.start_run(run_name=f"RF_n{n_est}_d{max_d}"):
            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", max_d)
            
            model, r2 = train_single_model("RandomForest", X_train, y_train, X_test, y_test,
                                         n_estimators=n_est, max_depth=max_d)
            
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(model, "model")
            print(f"  RF n{n_est}_d{max_d}: R²={r2:.4f}")
    
    # XGB sweep
    for n_est, lr in itertools.product(xgb_params["n_estimators"], xgb_params["learning_rate"]):
        with mlflow.start_run(run_name=f"XGB_n{n_est}_lr{lr}"):
            mlflow.log_param("model", "XGBRegressor")
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("learning_rate", lr)
            
            model, r2 = train_single_model("XGBRegressor", X_train, y_train, X_test, y_test,
                                         n_estimators=n_est, learning_rate=lr)
            
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(model, "model")
            print(f"  XGB n{n_est}_lr{lr}: R²={r2:.4f}")
    
    # Baseline Linear (no hyperparams)
    with mlflow.start_run(run_name="LinearRegression_baseline"):
        mlflow.log_param("model", "LinearRegression")
        model, r2 = train_single_model("LinearRegression", X_train, y_train, X_test, y_test)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
        print(f"  LinearRegression: R²={r2:.4f}")
    
    print("✅ Multi-model sweep complete! 16 runs logged!")
    print("→ MLflow UI: Charts → Parallel Coordinates → Model comparison!")

if __name__ == "__main__":
    main()
