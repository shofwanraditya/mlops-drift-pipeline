# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

def load_and_prepare_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Simpan dataset awal sebagai baseline/reference data
    os.makedirs("data", exist_ok=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/reference.csv", index=False)
    return train_df, test_df

def train_model(train_data: pd.DataFrame, test_data: pd.DataFrame, run_name: str = "initial_run"):
    target_col = "MedHouseVal"
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    mlflow.set_experiment("Automated_Housing_Price_Predictor")
    
    with mlflow.start_run(run_name=run_name) as run:
        n_estimators = 100
        max_depth = 12
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Logging parameter dan metrik ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Simpan artifact model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/production_model.joblib")
        
        # Registrasi model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Housing_Production_Model"
        )
        
        print(f"[{run_name}] Training Selesai | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        return run.info.run_id

if __name__ == "__main__":
    train_df, test_df = load_and_prepare_data()
    train_model(train_df, test_df, run_name="Baseline_Model")