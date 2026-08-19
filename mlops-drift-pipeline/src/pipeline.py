# src/pipeline.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.drift_detector import check_data_drift
from src.train import train_model

REFERENCE_DATA = "data/reference.csv"
CURRENT_DATA = "data/production_batch.csv"

def run_pipeline():
    print(">>> [MLOps Pipeline] Memulai inspeksi data produksi...")
    
    if not os.path.exists(CURRENT_DATA):
        print("Data produksi tidak ditemukan. Pipeline dihentikan.")
        return
    
    # 1. Jalankan Uji Data Drift
    drift_result = check_data_drift(REFERENCE_DATA, CURRENT_DATA, drift_share_threshold=0.25)
    print(f">>> Hasil Uji Drift: {drift_result}")
    
    if not drift_result["retrain_triggered"]:
        print("Distribusi data masih stabil. Tidak diperlukan retraining.")
        return
    
    print(" PERINGATAN: Data Drift Terdeteksi! Memulai Continuous Retraining Pipeline...")
    
    # 2. Persiapkan Dataset Baru (Menggabungkan Reference + New Batch)
    ref_df = pd.read_csv(REFERENCE_DATA)
    curr_df = pd.read_csv(CURRENT_DATA)
    
    # Simulasi ground truth target untuk data batch produksi
    if "MedHouseVal" not in curr_df.columns:
        curr_df["MedHouseVal"] = ref_df["MedHouseVal"].sample(len(curr_df), replace=True).values
    
    combined_df = pd.concat([ref_df, curr_df], ignore_index=True)
    
    # 3. Retraining Model dengan Data Terkini
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    run_id = train_model(train_df, test_df, run_name="Continuous_Retrain_After_Drift")
    
    # 4. Update Reference Baseline ke Dataset Gabungan
    combined_df.to_csv(REFERENCE_DATA, index=False)
    
    print(f" Continuous Training Selesai. Model terdaftar di MLflow Run ID: {run_id}")
    print(" Laporan drift visual tersimpan di 'data/drift_report.html'")

if __name__ == "__main__":
    run_pipeline()