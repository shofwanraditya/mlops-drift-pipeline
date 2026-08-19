# test_simulation.py
import pandas as pd
import numpy as np
import os
from src.pipeline import run_pipeline

def generate_drifted_batch():
    """Membuat data batch baru dengan pergeseran statistik buatan."""
    ref_df = pd.read_csv("data/reference.csv").drop(columns=["MedHouseVal"])
    batch_size = 500
    sample = ref_df.sample(n=batch_size, random_state=123).copy()
    
    # Injeksi Data Drift: Menggeser mean dan varians fitur
    sample["MedInc"] = sample["MedInc"] * 2.5 + np.random.normal(2.0, 0.5, size=batch_size)
    sample["AveRooms"] = sample["AveRooms"] * 1.8
    sample["HouseAge"] = sample["HouseAge"] * 0.4
    
    sample.to_csv("data/production_batch.csv", index=False)
    print(f"[Simulasi] {batch_size} sample data produksi ter-drift berhasil digenerate.")

if __name__ == "__main__":
    # 1. Generate data yang mengalami drift
    generate_drifted_batch()
    
    # 2. Jalankan pipeline orchestrator
    run_pipeline()