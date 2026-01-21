# src/gender_classification/model_trainer.py
# 🔥 MULTI-CLASS ROC-AUC FIXED

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from scipy.stats import randint, loguniform

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'gender_classification', 'model.pkl')
    model_metrics_file_path: str = os.path.join('artifacts', 'gender_classification', 'model_metrics.json')
    best_model_name_file_path: str = os.path.join('artifacts', 'gender_classification', 'best_model_name.json')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.label_encoder = None

    def load_artifacts(self, label_encoder_path):
        """Load label encoder"""
        self.label_encoder = load_object(label_encoder_path)
        logging.info("✅ Label encoder loaded")
        logging.info(f"🏷️  Classes: {self.label_encoder.classes_}")

    def safe_roc_auc(self, y_true, y_proba):
        """✅ FIXED: Multi-class ROC-AUC"""
        try:
            if y_proba.shape[1] == 2:  # Binary
                return roc_auc_score(y_true, y_proba[:, 1])
            else:  # Multi-class
                return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            return None

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """✅ MULTI-CLASS SAFE EVALUATION"""
        try:
            cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            y_pred = model.predict(X_test)
            
            # ✅ FIXED ROC-AUC for multi-class
            y_proba = model.predict_proba(X_test)
            roc_auc = self.safe_roc_auc(y_test, y_proba)
            
            metrics = {
                'cv_f1_mean': float(cv_f1.mean()),
                'cv_f1_std': float(cv_f1.std()),
                'test_accuracy': float(accuracy_score(y_test, y_pred)),
                'test_f1_macro': float(f1_score(y_test, y_pred, average='macro')),
                'test_roc_auc': float(roc_auc) if roc_auc is not None else None
            }
            
            print(f"\n=== {model_name} ===")
            print(f"CV F1 (macro): {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
            print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
            print(f"Test F1 (macro): {metrics['test_f1_macro']:.3f}")
            if roc_auc:
                print(f"Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
            print("\nClassification report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
            
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """✅ Logistic Regression - Multi-class safe"""
        try:
            logging.info("🚀 Logistic Regression...")
            
            logreg_base = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
            logreg_base.fit(X_train, y_train)
            baseline_metrics = self.evaluate_model(logreg_base, X_train, y_train, X_test, y_test, "Baseline Logistic Regression")
            
            param_dist = {
                'C': loguniform(0.01, 100),
                'penalty': ['l2'],  # l1 doesn't work well with multi_class='ovr'
                'solver': ['lbfgs', 'liblinear']
            }
            
            logreg_search = RandomizedSearchCV(
                LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'), 
                param_distributions=param_dist, n_iter=20, scoring='f1_macro', 
                cv=5, random_state=42, n_jobs=-1
            )
            logreg_search.fit(X_train, y_train)
            
            tuned_metrics = self.evaluate_model(
                logreg_search.best_estimator_, X_train, y_train, X_test, y_test, "Tuned Logistic Regression"
            )
            
            return logreg_search.best_estimator_, {
                'baseline': baseline_metrics,
                'tuned': tuned_metrics,
                'best_params': logreg_search.best_params_,
                'best_cv_score': logreg_search.best_score_
            }
        except Exception as e:
            raise CustomException(e, sys)

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """✅ Random Forest - Multi-class native"""
        try:
            logging.info("🌲 Random Forest...")
            
            rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_base.fit(X_train, y_train)
            baseline_metrics = self.evaluate_model(rf_base, X_train, y_train, X_test, y_test, "Baseline Random Forest")
            
            rf_param_dist = {
                'n_estimators': randint(50, 200),
                'max_depth': [None, 5, 10],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2']
            }
            
            rf_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_dist, 
                n_iter=20, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1
            )
            rf_search.fit(X_train, y_train)
            
            tuned_metrics = self.evaluate_model(
                rf_search.best_estimator_, X_train, y_train, X_test, y_test, "Tuned Random Forest"
            )
            
            return rf_search.best_estimator_, {
                'baseline': baseline_metrics,
                'tuned': tuned_metrics,
                'best_params': rf_search.best_params_,
                'best_cv_score': rf_search.best_score_
            }
        except Exception as e:
            raise CustomException(e, sys)

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """✅ Gradient Boosting - Multi-class native"""
        try:
            logging.info("⚡ Gradient Boosting...")
            
            gb_base = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb_base.fit(X_train, y_train)
            baseline_metrics = self.evaluate_model(gb_base, X_train, y_train, X_test, y_test, "Baseline Gradient Boosting")
            
            gb_param_dist = {
                'learning_rate': loguniform(0.01, 0.3),
                'n_estimators': randint(50, 200),
                'max_depth': [3, 5],
                'min_samples_leaf': randint(10, 50)
            }
            
            gb_search = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=42), gb_param_dist, 
                n_iter=20, cv=5, scoring='f1_macro', random_state=42, n_jobs=-1
            )
            gb_search.fit(X_train, y_train)
            
            tuned_metrics = self.evaluate_model(
                gb_search.best_estimator_, X_train, y_train, X_test, y_test, "Tuned Gradient Boosting"
            )
            
            return gb_search.best_estimator_, {
                'baseline': baseline_metrics,
                'tuned': tuned_metrics,
                'best_params': gb_search.best_params_,
                'best_cv_score': gb_search.best_score_
            }
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr_path, test_arr_path, label_encoder_path):
        """🚀 Main method - MULTI-CLASS SAFE"""
        try:
            logging.info("🎯 Multi-class model training START")
            
            train_arr = np.load(train_arr_path)['arr']
            test_arr = np.load(test_arr_path)['arr']
            
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
            
            logging.info(f"📊 Data: X_train={X_train.shape}, classes={len(np.unique(y_train))}")
            
            self.load_artifacts(label_encoder_path)
            
            models_results = {}
            
            logreg_model, logreg_results = self.train_logistic_regression(X_train, y_train, X_test, y_test)
            models_results['LogisticRegression'] = logreg_results
            
            rf_model, rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
            models_results['RandomForest'] = rf_results
            
            gb_model, gb_results = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
            models_results['GradientBoosting'] = gb_results
            
            # Best model selection
            best_model_name = max(models_results.keys(), 
                                key=lambda k: models_results[k]['tuned']['test_f1_macro'])
            best_model_f1 = models_results[best_model_name]['tuned']['test_f1_macro']
            
            model_mapping = {
                'LogisticRegression': logreg_model,
                'RandomForest': rf_model,
                'GradientBoosting': gb_model
            }
            best_model = model_mapping[best_model_name]
            
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"   Test F1 Score: {best_model_f1:.3f}")
            
            # Save
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            
            all_results = {
                'best_model': best_model_name,
                'best_f1_score': float(best_model_f1),
                'all_models': models_results
            }
            
            with open(self.model_trainer_config.model_metrics_file_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logging.info("🎉✅ MULTI-CLASS TRAINING COMPLETE!")
            return best_model_f1
            
        except Exception as e:
            raise CustomException(e, sys)
