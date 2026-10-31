import json
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from src.features import engineer_churn_metrics

# 1. Dynamically locate the directory where THIS file (features.py) physically lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build a reliable, absolute path to config.json relative to features.py
# (This goes up one folder from src/ to the root, then enters config/)
CONFIG_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "config", "config.json"))

# Read your Central project blueprint config
with open(CONFIG_PATH, "r") as config_file:
    config=json.load(config_file)
def get_production_pipeline() -> Pipeline:
    
    # 1. Extract feature arrays dynamically
    robust_features = config["preprocessing_parameters"]["robust_scaler_features"]
    standard_features = config["preprocessing_parameters"]["standard_scaler_features"]
    categorical_features = config["data_schemas"]["categorical_features"]

    # 2. Construct the ColumnTransformer routing map
    preprocessor = ColumnTransformer(
        transformers=[
            ("robust_scale", RobustScaler(), robust_features),  
            ("standard_scale", StandardScaler(), standard_features),  
            ("categorical_passthrough", "passthrough", categorical_features)
        ],
        remainder="drop",  
    )

    # 4. Chain everything together into a unified Scikit-Learn Pipeline object
    production_pipeline = Pipeline(
        steps=[("custom_feature_engineering", FunctionTransformer(engineer_churn_metrics)),
               ("core_preprocessing_transformers", preprocessor)]
    )

    return production_pipeline
