# Customer Churn Prediction System

Developed as the second practical project during my Machine Learning Engineering Internship at IncodeVision. This repository houses a production-grade, schema-driven predictive pipeline designed to calculate future subscriber exit trajectories and flag critical retention risks.

Instead of deploying a monolithic, hardcoded script workbook, this architecture strictly decouples system configuration, custom feature engineering, data-leakage firewalls, and deployment interfaces into modular, production-ready modules.

# Dataset Architecture

The system is built and optimized using the open-source UCI Iranian Telecom Churn Dataset (3,150 rows, 14 dimensional properties). It captures real-world telecom customer usage behaviors, account timelines, and customer friction indicators over a 9-month operational window.

• Target Label Imbalance: ~84.3% Active/Loyal (0), ~15.7% Churned (1).

• Data Context: Free from arbitrary missing string values, but heavily saturated with right-skewed tracking distributions and extreme operational outliers representing power-user accounts.

# Production Directory Structure

```bash

customer-churn-prediction/

├── config/

│   └── config.json                     # Locked system configurations and data schema

├── data/

│   ├── raw_data/

│      └── Customer Churn.csv          # Unmodified source dataset

│   

├── notebooks/

│   ├── 01_eda_and_sweetviz.ipynb       # Statistical visualizations & UI profiles

│   ├── 02_feature_selection.ipynb      # The 5-layer programmatic pruning pipeline

│   └── raw_customer_churn_profile.html # Static offline Sweetviz diagnostic report

├── src/

│   ├── __init__.py                     # Package initialization marker

│   ├── features.py                     # Isolated math logic for engineered metrics

│   ├── pipeline.py                     # Data-leakage firewall ColumnTransformer layout

│   └── train.py                        # Hyperparameter grid search & threshold tuning loop

├── artifacts/

│   ├── preprocessor.joblib             # Saved fitted scaling/engineering pipeline binary

│   └── churn_model.pkl                 # Final optimized Random Forest Classifier binary

├── app.py                              # Unified Streamlit deployment dashboard interface

└── requirements.txt                    # Clean isolated environment dependency mapping

```

# Unified Core Engineering Workflows

1. Programmatic 5-Layer Notebook Pruning Pipeline

Before training, features pass through sequential mathematical filters to ensure low model latency and absolute generalization strength:

• The Leakage Firewall: Immediate stratified train-test splitting occurs before scaling, computing correlations, or checking relationships to prevent validation biasses.

• Pass A (High-Cardinality Filter): Strips high-cardinality values where distinct mappings scale past 95% of row counts (e.g. system keys).

• Pass B (Stagnant Value Scrubber): Removes categorical or continuous rows where a single value dominates over 98% of the records.

• Pass C (Information Entropy Leakage Guard): Screens remaining features via automated Mutual Information scoring. Features scoring above 0.50 are flagged as post-event target proxies (e.g. Status) and structurally dropped.

• Pass D (Collinearity Purge): Computes a Spearman Rank matrix on continuous inputs. If features share a lockstep trajectory exceeding 0.85 (e.g., Seconds of Use vs Frequency of use), the lower-level aggregate is pruned.

• Pass E (Outlier Routing Audit): Evaluates skewness across remaining fields. If skewness magnitude scales past |0.75|, the column is dynamically routed to an internal RobustScaler (Median/IQR mapping) inside config.json instead of getting truncated or discarded.

2. Schema-Driven Deployment Design

Every operational step inside src/pipeline.py and app.py is entirely dynamic. The scripts contain no hardcoded column names or list definitions. They ingest arrays directly from config/config.json at runtime. If you add a custom calculation into src/features.py (like SMS_to_Voice_Ratio) and append its tag name to the JSON schema, the downstream estimators and scalers absorb it automatically.

3. Business-First Threshold Tuning

Instead of blindly accepting scikit-learn's standard default 0.50 decision threshold, the training script computes a Precision-Recall Curve on the validation split. It locks onto the precise decimal cutoff that maximizes the baseline F1-Score, ensuring the business perfectly balances its preventative marketing budget against missed customer churn flags.

# Installation & Local Execution

Ensure you are operating inside a clean, isolated virtual environment before spinning up the local web engine.

1. Install Core System Dependencies:

```bash

pip install -r requirements.txt

```

2. Execute the End-to-End Training & Serialization Pipeline:

```bash

$env:PYTHONPATH="." ; python src/train.py

```

(This reads the dynamic json layout, executes a 5-fold cross-validation GridSearchCV across 180 tree hyperparameter states, tunes the PR-curve threshold, prints the classification reports, and writes fresh serialized binaries to artifacts/)

3. Launch the Real-Time Streamlit Prediction Interface:

```bash

streamlit run app.py

```



