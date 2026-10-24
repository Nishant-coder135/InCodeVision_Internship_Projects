import pandas as pd
import numpy as np
import re
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Ingesting payloads
vectorizer=joblib.load(r"C:\Users\hp\OneDrive\Desktop\InCodeVision Internship Projects\Project-01\Spam Email Classifier\model\text_vectorizer.joblib")
model_payloads=joblib.load(r"C:\Users\hp\OneDrive\Desktop\InCodeVision Internship Projects\Project-01\Spam Email Classifier\model\spam_classifier.joblib")
classifier=model_payloads['classifier']

# Convert library stop_words into set
stop_words=set(ENGLISH_STOP_WORDS)

# Text Cleaning Robust Function
def clean_raw_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    
    lowercase_text=raw_text.lower()
    stripped_text=re.sub(r'[^a-zA-Z\s]', '', lowercase_text)
    tokens=stripped_text.split()
    meaningful_words=[word for word in tokens if word not in stop_words]
    return " ".join(meaningful_words)

st.title("Spam Email Classifier")
st.write("Type or paste any message below to check if it is Spam or Legitimate.")
user_input=st.text_area("Enter Email here : ", placeholder="Type here .......")

if st.button("Classify Email"):
    if user_input.strip()=="":
        st.warning("Please type some text first!!")
    else:
        cleaned_text=clean_raw_text(user_input)
        text_vectorized=vectorizer.transform([cleaned_text])
        probabilities=classifier.predict_proba(text_vectorized)
        probability_index=np.argmax(probabilities)
        confidence_score=(probabilities[0][probability_index])*100
        if probability_index==1:
            st.error(f"This is Spam! (Confidence Score : {confidence_score:.2f}%)")
        else:
            st.success(f"This is Legitimate! (Confidence Score : {confidence_score:.2f}%)")