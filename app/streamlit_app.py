import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# Load models
tokenizer = BertTokenizer.from_pretrained("../models/sentiment-bert")
classifier = BertForSequenceClassification.from_pretrained("../models/sentiment-bert")
summarizer = pipeline("summarization", model="t5-small")

st.title("🧠 MediBERTx - Clinical NLP Toolkit")

text_input = st.text_area("Paste a clinical note here:", height=200)

if st.button("Classify"):
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = classifier(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    st.success(f"Predicted Class: {'Positive' if predicted_class == 1 else 'Negative'}")

if st.button("Summarize"):
    summary = summarizer(text_input, max_length=60, min_length=20, do_sample=False)
    st.markdown(f"**Summary:** {summary[0]['summary_text']}")