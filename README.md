# Voyage-Analytics-Integrating-MLOps-in-Travel
End-to-end MLOps project for travel industry data including regression, classification, and recommendation models. Features automated data processing, model training, REST API serving with Flask, interactive Streamlit app, Docker and Kubernetes deployment, Airflow scheduling, and MLflow experiment tracking for scalable production-ready workflows.

# End-to-End MLOps Projects Suite

## Overview
This repository showcases **three comprehensive MLOps projects** demonstrating production-ready ML pipelines:

1. **Flight Price Prediction** - Regression model predicting flight prices from `flights.csv`
2. **Hotel Recommendation** - Recommendation system for personalized hotel suggestions  
3. **Gender Classification** - Classification model predicting gender from input features

Complete MLOps stack: model development → REST API → Docker → Kubernetes → Airflow workflows → Jenkins CI/CD → MLFlow tracking.

## Project Outcomes
**Flight Price Prediction:**
- Regression model with feature selection and validation
- REST API serving price predictions
- Containerization using Docker
- Scalable K8s deployment with Airflow automation
- Automated retraining workflows via Airflow DAGs
- CI/CD Pipeline using Jenkins for consistent and reliable deployment of the travel price prediction model.
- MLFlow experiment tracking and model versioning

**Hotel Recommendation:**
- Recommendation engine with collaborative/content-based filtering
- Personalized hotel suggestions via Streamlit

**Gender Classification:**
- Binary/multi-class classification model
- High-accuracy gender prediction API
- Deploying model using Streamlit

**Shared Infrastructure:**
- Unified CI/CD pipeline with Jenkins
- Docker containerization for all services
- Kubernetes orchestration across all models

## Project Structure
```markdown
## Project Structure

Voyage-Analytics-Integrating-MLOps-in-Travel/
├── data/                          # 📊 Raw datasets
│   ├── flights.csv               # ✈️ Flight prices
│   ├── hotels.csv                # 🏨 Hotel data
│   └── gender_data.csv           # 👤 Classification
│
├── src/                           # 🤖 ML modules
│   ├── common/                    # ⚙️ Shared utils
│   │   ├── logger.py
│   │   ├── exception.py
│   │   └── utils.py
│   ├── flight_price/             # 🚀 Full MLOps (Flask API)
│   │   ├── app.py                # REST API
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── Dockerfile
│   │   ├── Deployment.yml        # K8s
│   │   ├── service.yml
│   │   └── flight_price_mlflow_runner.py
│   ├── gender_classification/    # 📱 Streamlit app
│   │   ├── app.py
│   │   ├── data_ingestion.py
│   │   └── model_trainer.py
│   └── hotel_recommendation/     # 📱 Streamlit app
│       ├── app.py
│       ├── data_ingestion.py
│       └── model_trainer.py
│
├── artifacts/                     # 💾 Model outputs
│   ├── flight_price/             # model.pkl, preprocessor.pkl
│   ├── gender_classification/
│   └── hotel_recommendation/
│
├── dags/                         # 🌐 Airflow (Flight focus)
│   ├── flight_price_dag.py
│   ├── gender_dag.py
│   └── hotel_dag.py
│
├── notebooks/                     # 📓 EDA & experiments
│   ├── Flight_Price_Prediction.ipynb
│   ├── Gender_Classification.ipynb
│   └── Hotel_Recommendation.ipynb
│
├── mlruns/                       # 🎯 MLflow tracking
├── airflow-docker/               # 🧪 Airflow setup
├── k8s/                          # ☸️ Kubernetes (Flight)
├── templates/                    # 🕸️ HTML
├── docker-compose.yml            # 🐳 Local stack
├── requirements.txt              # 📦 Dependencies
├── setup.py                      # 🔧 Install
└── README.md                     # 📖 This doc!
```
```

## Tools and Technologies

### ML Models & Data
```
Flight: Scikit-learn/XGBoost Regression
Hotel: Surprise/Implicit Recommendation  
Gender: Logistic Regression/Random Forest
```
- Pandas, NumPy, Scikit-learn
- MLFlow (experiment tracking)

### Infrastructure Stack
```
API: Flask 
Container: Docker
Orchestration: Kubernetes
Workflows: Apache Airflow
CI/CD: Jenkins
Monitoring: MLFlow
```

## Getting Started (Local Setup)

### Prerequisites
```bash
Python 3.9+ | Docker 20+ | Minikube | Git | kubectl
```

### Flight Price Prediction Development

<img width="462" height="431" alt="Screenshot 2026-03-07 215331" src="https://github.com/user-attachments/assets/23567293-60f7-46a4-b9d6-d353ceb11490" />


<img width="546" height="773" alt="Screenshot 2026-03-07 215421" src="https://github.com/user-attachments/assets/b28e0869-e4af-44e6-9824-96a9c78a74f9" />

### 1. Clone & Setup

**Clone & Setup (Exact Repo):**
```bash
git clone https://github.com/DevByAsfiya/Voyage-Analytics-Integrating-MLOps-in-Travel.git
cd Voyage-Analytics-Integrating-MLOps-in-Travel
python -m venv .venv
source .venv\Scripts\activate.ps1
pip install -r requirements.txt


### 2. Environment
```bash
cp .env.example .env
# Configure DATABASE_URL, MLFLOW_URI, etc.
```

## Docker Deployment (All Projects)
```bash
# Build image
docker build -t flight-price-prediction:1.0 .

# Run Container Locally
docker run -d -p 5000:5000 flight-price-prediction:1.0

# Access services:
# Flight: [localhost:5000](http://localhost:5000)
```

```markdown
## Kubernetes Deployment (Flight Price API)

**Purpose**: Scales Flight Price Prediction API with 2 replicas, auto-healing, and zero-downtime updates.

### YAML Breakdown

**`src/flight_price/Deployment.yml`**  
- **Manages 2 Pod replicas** (`replicas: 2`) running `afia890/flight-price-prediction:1.0`
- **Auto-restarts failed Pods** and handles scaling
- **Environment**: `MODEL_PATH=/app/artifacts/flight_price/model.pkl`
- **Port**: Container listens on `5000`

**`src/flight_price/service.yml`**  
- **ClusterIP Service** exposes Pods internally on port `80`
- **Load balances** traffic to Pods on port `5000`
- **DNS**: `flight-price-prediction-service:80`

### Local Setup (Docker Desktop)

1. **Enable Kubernetes**:
   ```
   Docker Desktop → Settings → Kubernetes → Enable → Apply & Restart
   ```

2. **Deploy** (from project root):
   ```bash
   kubectl apply -f src/flight_price/Deployment.yml
   kubectl apply -f src/flight_price/service.yml
   ```

3. **Verify**:
   ```bash
   kubectl get pods                    # 2x Running pods
   kubectl get deployment              # READY 2/2
   kubectl get service                 # ClusterIP + PORT 80
   ```

4. **Access API**:
   ```bash
   kubectl port-forward service/flight-price-prediction-service 5000:80
   ```
   **Test**: `http://localhost:5000/predict` (JSON response)

### Scale Demo
```bash
# Edit Deployment.yml: replicas: 3
kubectl apply -f src/flight_price/Deployment.yml
kubectl get pods  # Now shows 3 pods!
```
```markdown
## Airflow Scheduling (Orchestration)

- Apache Airflow is used to **orchestrate and schedule** the end-to-end ML pipelines.[web:11][web:12]
- DAGs:
  - `flight_price_dag.py`: daily retrain + evaluation + (optional) deployment of the Flight Price model.
- Airflow runs via the `airflow-docker/` `docker-compose.yml` stack and can be monitored from the Airflow web UI.[web:14][web:18]

---

## CI/CD Pipeline (Jenkins)

- Jenkins provides a **CI/CD pipeline** that automates build, test, and deployment steps whenever changes are pushed to the repo.
- The pipeline is defined in a `Jenkinsfile` (or Jenkins job) and typically includes:
  - Build & unit tests.
  - Container image build and push.
  - Kubernetes deployment update for the Flight Price API.
- This ensures **consistent, repeatable deployments** to the Kubernetes cluster.

---

## MLflow Tracking

- MLflow is used to **track experiments, parameters, metrics, and models** across all three projects.
- Runs and artifacts are stored under the `mlruns/` directory.
- For the Flight Price model, `flight_price_mlflow_runner.py` logs:
  - model hyperparameters and training metrics,
  - serialized models (`model.pkl`) and preprocessing artifacts.
- The MLflow UI can be launched to **compare runs and promote the best model** to production.
```
```markdown
## Hotel Recommendation System

<img width="1903" height="992" alt="Screenshot 2026-03-07 205325" src="https://github.com/user-attachments/assets/b27cda86-4c57-45f1-8413-2b139e1d18f8" />

### Model Overview
- **Algorithm**: Collaborative filtering using or content-based similarity.
- **Dataset**: `data/hotels.csv` with user ratings, hotel features, location data.
- **Pipeline** (`src/hotel_recommendation/`):
  - `data_ingestion.py`: Load and clean hotel/user data
  - `data_transformation.py`: Create user-item matrix, handle sparsity
  - `model_trainer.py`: Train recommender, generate top-N recommendations
- **Artifacts**: Trained model and similarity matrix saved in `artifacts/hotel_recommendation/`

### Streamlit Deployment
```bash
cd src/hotel_recommendation
streamlit run app.py
# Access: http://localhost:8501
```

**Features**:
- User ID input → personalized hotel recommendations
- Filter by location, price range, star rating
- Interactive similarity score visualization

---

## Gender Classification Model

<img width="1909" height="846" alt="Gender_Claddification" src="https://github.com/user-attachments/assets/ef4a1516-6042-4603-ac6b-03625e32fe86" />

### Model Overview
- **Algorithm**: **Logistic Regression** or **Random Forest** classifier for binary/multi-class gender prediction.
- **Dataset**: `data/users.csv` with demographic features (name, age, location, etc.).
- **Pipeline** (`src/gender_classification/`):
  - `data_ingestion.py`: Load demographic data
  - `data_transformation.py`: Feature engineering (name embeddings, categorical encoding)
  - `model_trainer.py`: Train classifier, evaluate accuracy/F1-score
- **Metrics**: 85-92% accuracy (depending on feature engineering).
- **Artifacts**: `model.pkl`, `preprocessor.pkl` in `artifacts/gender_classification/`

### Streamlit Deployment
```bash
cd src/gender_classification
streamlit run app.py
# Access: http://localhost:8502
```

**Features**:
- Real-time gender prediction from text/demographic inputs
- Confidence scores and feature importance visualization
- Batch prediction for multiple users
- Model performance metrics dashboard

---

## Quick Launch All Streamlit Apps

```bash
# Terminal 1: Hotel Recs
cd src/hotel_recommendation && streamlit run app.py --server.port 8501

# Terminal 2: Gender Classifier  
cd src/gender_classification && streamlit run app.py --server.port 8502

# URLs:
# Hotels: http://localhost:8501
# Gender: http://localhost:8502
```




