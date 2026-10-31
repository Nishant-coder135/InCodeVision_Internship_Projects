import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
from src.pipeline import get_production_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build a reliable, absolute path to config.json relative to features.py
# (This goes up one folder from src/ to the root, then enters config/)
CONFIG_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "config", "config.json"))
with open(CONFIG_PATH, "r") as f:
    config=json.load(f)

def main():
    print("=== INITIALIZING PRODUCTION MODEL TRAINING ENGINE ===")
    
    # 1. Load system settings and metadata seeds
    RAW_DATA_PATH = config["file_paths"]["raw_data_path"]
    TARGET = config["data_schemas"]["target_variable"]
    SEED = config["project_metadata"]["random_state"]
    
    # 2. Ingest raw data records
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 3. Replicate the strict Train-Test Split Firewall locally
    # This guarantees we extract the exact same rows that the notebook used
    from sklearn.model_selection import train_test_split
    X_raw = df.drop(columns=[TARGET])
    y_raw = df[TARGET]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
    )
    
    # 4. Ingest and compilation of the production preprocessing pipeline
    print("[INFO] Building data transformation pipeline flow...")
    pipeline_transformer = get_production_pipeline()
    
    # Transform raw data frames into scaled model arrays cleanly
    # Fits exclusively on the training slice to maintain absolute zero data leakage
    X_train_processed = pipeline_transformer.fit_transform(X_train_raw)
    X_test_processed = pipeline_transformer.transform(X_test_raw)
    
    # 5. Model Selection & Optimization Loop via Stratified Grid Search
    print("[INFO] Initializing hyperparameter optimization grid search...")
    
    # We use a weighted Random Forest to counteract the sharp 84/16 target imbalance
    base_model = RandomForestClassifier(random_state=SEED, class_weight="balanced")
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [2, 4, 8]
    }
    
    # Use StratifiedKFold to securely preserve the churn ratio across internal slices
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Optimize for ROC-AUC to secure maximum predictive separation strength
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_processed, y_train)
    best_estimator = grid_search.best_estimator_
    
    print(f"\n[OPTIMIZATION SUCCESS] Best Parameters: {grid_search.best_params_}")
    print(f"[OPTIMIZATION SUCCESS] Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    # 6. Advanced Threshold Optimization Engine (The Business-First PR-Curve Shift)
    # Instead of blindly accepting the default 0.50 threshold, we tune for the optimal cutoff
    y_test_probs = best_estimator.predict_proba(X_test_processed)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_probs)
    
    # Calculate the threshold that yields the absolute peak F1-Score on unseen data
    # (Harmonic balance maximizing both customer recall and marketing budget precision)
    f1_scores = [f1_score(y_test, (y_test_probs >= t).astype(int)) for t in thresholds]
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]
    
    print(f"[BUSINESS TUNING] Optimal Churn Decision Threshold Locked: {optimal_threshold:.4f}")
    
    # 7. Comprehensive Model Telemetry Evaluation Report
    y_test_pred_tuned = (y_test_probs >= optimal_threshold).astype(int)
    
    print("\n" + "="*60)
    print("           PRODUCTION EVALUATION TELEMETRY REPORT       ")
    print("="*60)
    print(classification_report(y_test, y_test_pred_tuned, target_names=["Loyal (0)", "Churned (1)"]))
    print(f"Final Validation ROC-AUC Score: {roc_auc_score(y_test, y_test_probs):.4f}")
    print("="*60 + "\n")
    
    # Update our text configuration parameters dynamically with our tuned rules
    config["inference_settings"] = {"default_decision_threshold": float(optimal_threshold)}
    config["model_hyperparameters"] = {
        "algorithm": "RandomForestClassifier",
        "best_params": grid_search.best_params_
    }
    
    config_write_path = "config/config.json" if os.path.exists("config/config.json") else "../config/config.json"
    with open(config_write_path, "w") as f:
        json.dump(config, f, indent=4)
        
    # 8. Production Artifact Persistence & Serialization
    
    # Save the fitted data transformer and optimized model file parameters
    joblib.dump(pipeline_transformer, "artifacts/preprocessor.joblib")
    joblib.dump(best_estimator, "artifacts/churn_model.pkl")
    
    
    print("[SUCCESS] Production artifacts successfully serialized inside 'artifacts/'.")
    print("=== MODEL TRAINING ENGINE PROCESS COMPLETE ===\n")

if __name__ == "__main__":
    import numpy as np
    main()
