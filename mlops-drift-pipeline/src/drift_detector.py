# src/drift_detector.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_data_drift(reference_path: str, current_path: str, drift_share_threshold: float = 0.3) -> dict:
    """
    Memeriksa data drift antara baseline dan current data.
    drift_share_threshold: Jika rasio kolom yang ter-drift >= threshold, trigger retrain.
    """
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)
    
    # Hapus kolom target agar fokus pada distribusi fitur input
    if "MedHouseVal" in ref_df.columns:
        ref_df = ref_df.drop(columns=["MedHouseVal"])
    if "MedHouseVal" in curr_df.columns:
        curr_df = curr_df.drop(columns=["MedHouseVal"])
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)
    
    result = report.as_dict()
    
    dataset_drift = False
    share_drifted_features = 0.0
    drifted_features = []
    
    # Iterasi seluruh metrik yang dihasilkan oleh DataDriftPreset
    for metric in result.get("metrics", []):
        metric_res = metric.get("result", {})
        
        # Ekstraksi status ringkasan dataset drift
        if "dataset_drift" in metric_res:
            dataset_drift = metric_res.get("dataset_drift", False)
            share_drifted_features = metric_res.get("share_of_drifted_columns", 0.0)
            
        # Ekstraksi daftar kolom spesifik yang mengalami drift
        if "drift_by_columns" in metric_res:
            cols_dict = metric_res.get("drift_by_columns", {})
            drifted_features = [
                col for col, data in cols_dict.items() if data.get("drift_detected", False)
            ]
    
    is_retrain_needed = share_drifted_features >= drift_share_threshold
    
    summary = {
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(share_drifted_features, 4),
        "drifted_features": drifted_features,
        "retrain_triggered": is_retrain_needed
    }
    
    # Simpan visual report HTML
    report.save_html("data/drift_report.html")
    
    return summary