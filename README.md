\# InCodeVision Machine Learning Internship Portfolio



Welcome to my repository tracking the core machine learning pipelines, predictive systems, and production web applications built during my internship at InCodeVision.



Rather than relying on default configuration templates, my focus throughout this internship was on engineering end-to-end, leakage-proof data preprocessing workflows, handling real-world dataset skews, optimizing classification boundaries, and deploying production-ready models as functional standalone software tools.



\-----------------------------------------------------------------------------------------------------------------------



\# Project Roadmap \& Deliverables



\# Project 1: Production-Grade Spam Email Classifier Web Application

* Core Focus: Natural Language Processing (NLP), text tokenization, and vector space transformations.
* Implementations: Designed a text cleansing function utilizing regular expressions ('\[^a-zA-Z\\s]') to strip punctuation, uppercase variance, and numeric noise. Implemented TF-IDF vectorization capped at 3,000 top vocabulary terms, fitting parameters strictly on training splits to prevent data leakage. Trained a Multinomial Naive Bayes classifier that accounts for the mild 76/24 dataset imbalance via built-in class priors.
* Key Metrics: 98.00% Global Accuracy | 0.98 Spam Precision | 0.94 Spam Recall (0.96 F1-Score).
* Deliverables: A standalone, fully interactive Streamlit web application ('app.py') that ingests raw user text blocks in real time, processes tokens through the identical cleaning pipeline, and extracts classification probabilities via '.predict\_proba()' to render dynamic confidence score cards.



\# Project 2: Customer Churn Prediction Engine \[Ongoing Project]

* Core Focus: Tabular behavioral analytics, feature interaction tracking, and leakage-proof preprocessing matrices.



\-----------------------------------------------------------------------------------------------------------------------



\# Repository Directory Layout



```text

InCodeVision\_Internship\_Projects/

│

├── .venv/                             # Unified Local Environment Container (Git Ignored)

├── .gitignore                         # Central Repository Resource Filter Rules

├── README.md                          # Global Portfolio Documentation Sheet

│

└── Project-01/

&#x20;   └── Spam Email Classifier/

&#x20;       ├── data/                      # Isolated Text Data Segment Storage

&#x20;       ├── model/                     # Serialized Pipeline Payload Directory

&#x20;       ├── notebook/                  # Operational Preprocessing \& Training Workspaces

&#x20;       ├── app.py                     # Streamlit User Interface Script

&#x20;       └── requirements.txt           # Local Dependency Checklist

```



\-----------------------------------------------------------------------------------------------------------------------



\# Global Setup \& Running the Application



To safeguard against environment cross-contamination and ensure isolated execution, use your terminal to set up a unified root environment box before running the individual sub-projects:



\# 1. Clone the Portfolio Workspace

```bash

git clone https://github.com/Nishant-coder135/InCodeVision\_Internship\_Projects

cd InCodeVision\_Internship\_Projects

```



\# 2. Configure a Unified Virtual Environment

Initialize and activate a single, clean workspace environment container at the absolute root folder path:

```bash

\# Initialize the virtual environment directory structure

python -m venv .venv



\# Activate the isolated environment layer (Windows OS)

.venv\\Scripts\\activate

```



\# 3. Install Sub-Project Specific Dependencies

Navigate directly into the target project folder and run the localized installer to download the precise libraries required for that specific build:

```bash

cd "Project-01/Spam Email Classifier"

pip install -r requirements.txt

```



\# 4. Launch the Live Streamlit Web Interface

Run the localized user interface dashboard code locally on your high-speed development server:

```bash

streamlit run app.py

```

This command spins up a native server loop and immediately opens the completed interactive application inside your web browser window.



