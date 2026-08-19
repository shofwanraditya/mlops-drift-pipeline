# MLOps Pipeline: Data Drift Detection & Automated Retraining

A practical MLOps pipeline designed to handle covariate shift (data drift) in tabular machine learning models. The system serves predictions via FastAPI, logs incoming traffic, checks for statistical distribution drift using Evidently AI, and triggers continuous retraining with MLflow experiment tracking.

## Overview

When machine learning models run in production, changes in real-world user behavior or environment often cause input distributions to shift (data drift), degrading model accuracy over time.

This project implements an automated feedback loop:
1. **Serving & Logging:** FastAPI serves real-time inferences and buffers incoming feature payloads.
2. **Drift Monitoring:** Statistical tests (Wasserstein distance / K-S test) compare live batches against baseline training data.
3. **Automated Retraining:** If more than 25% of features show significant drift, a retraining script runs automatically, evaluates the new model, and registers the run in MLflow.

[ Client ] ──(POST /predict)──▶ [ FastAPI ] ──▶ [ Log Features ]
                                                       │
                                                       ▼
[ MLflow Registry ] ◀── [ Retrain ] ◀── [ Drift Check (Evidently) ]







## Results & Screenshots

### 1. Data Drift Report (Evidently AI)
Statistical distribution drift detected on manipulated features (`MedInc`, `AveRooms`, `HouseAge`):

![Data Drift Report](assets/drift_report.png)

### 2. Model Tracking (MLflow UI)
Comparison of metrics ($RMSE$ and $R^2$) between the baseline model and the continuous retrained model:

![MLflow Comparison](assets/mlflow_comparison.png)

## Tech Stack

- **Model:** Scikit-learn (RandomForestRegressor), Pandas, NumPy
- **MLOps & Tracking:** MLflow, Evidently AI
- **API:** FastAPI, Uvicorn
- **Serialization:** Joblib

## Project Structure


mlops-drift-pipeline/
├── assets/                  # Screenshot bukti visual
├── data/
│   ├── reference.csv        # Dataset baseline
│   └── buffer.csv           # Buffer payload request
├── models/
│   └── production_model.joblib
├── src/
│   ├── train.py             # Script training baseline
│   ├── drift_detector.py    # Logika deteksi drift Evidently
│   ├── serve.py             # FastAPI prediction server
│   └── pipeline.py          # Orchestrator drift check + retrain
├── test_simulation.py       # Simulasi data drift
├── requirements.txt
├── .gitignore
└── README.md

HOW TO RUN
```text
1. Setup Environment

python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

2. Train Baseline Model
python -m src.train

3. Run Drift Simulation & Retraining
python test_simulation.py

4. View MLflow Dashboard
mlflow ui --port 5000
Open http://localhost:5000 to inspect experiment runs and compare metrics.

5. Run API Inference
uvicorn src.serve:app --reload --port 8000
Interactive documentation is available at http://localhost:8000/docs.

