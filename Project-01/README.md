\# Spam Email Classifier



An end-to-end, production-ready natural language processing (NLP) application designed to clean unstructured text streams, extract statistical word distributions via TF-IDF vectorization, and dynamically calculate spam metrics within a responsive user interface.



\---------------------------------------------------------------------------------------------------------------------------------



\# Data Pipeline \& Transformation Architecture



To maintain rigorous production standards and completely eliminate Data Leakage, all processing layers are systematically isolated:



\# 1. Linguistic Cleansing Pipeline

* Text Normalization: Converts raw email text characters to uniform lowercase fields to eliminate case variance.
* Regex Filtering: Deploys target regular expression algorithms ('\[^a-zA-Z\\s]') to completely strip out punctuation, continuous numbers, and digital emojis, preventing distance and token calculation distortions.
* Stopword Pruning: Utilizes standard English library dictionaries to drop common filler words, isolating highly descriptive keyword markers.



\# 2. Feature Vector Space (TF-IDF)

* Compresses text sequences into a 3,000-column numerical coordinate grid using Term Frequency-Inverse Document Frequency (TF-IDF) metrics.
* To safeguard partition integrity, the vectorization matrix is fit strictly on training segments, while the operational 'text\_vectorizer.joblib' object is saved to disk for live application routing.



\---------------------------------------------------------------------------------------------------------------------------------



\# Model Performance Diagnostics



The classifier was trained using a Multinomial Naive Bayes algorithm, leveraging dataset priors to handle text distributions efficiently. Due to mild class imbalances (76/24 split), global accuracy was bypassed to prioritize minority class tracking:



* Global Classification Accuracy: '98.00%'
* Spam Class (1) Precision: '0.98' (Minimizes false alarms, keeping clean emails safe)
* Spam Class (1) Recall: '0.94' (Intercepts 94% of incoming spam threats successfully)
* Spam Class (1) F1-Score: '0.96'



\---------------------------------------------------------------------------------------------------------------------------------



\# Live Streamlit Web Inference Application



The interactive user dashboard is decoupled from the training notebooks via 'app.py'. It dynamically ingests unstructured live text entries, processes them through the text cleaning function, converts vectors in real-time using the saved vocabulary weights, and extracts class probabilities via '.predict\_proba()' to display real-time metric cards.



\---------------------------------------------------------------------------------------------------------------------------------



\# Step-by-Step Interface Setup:



1\. Open your terminal window inside this specific project directory:

&#x20;  ```bash

&#x20;  cd "Project-01/Spam Email Classifier"

&#x20;  ```



2\. Ensure your global virtual environment container is active:

&#x20;  ```bash

&#x20;  # Execute from the root directory path if not active:

&#x20;  ..\\..\\.venv\\Scripts\\activate

&#x20;  ```



3\. Install the project specific dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Launch the live web application server:

&#x20;  ```bash

&#x20;  streamlit run app.py

&#x20;  ```

&#x20;  This will spin up a local development loop and host the user interface inside your web browser window.

