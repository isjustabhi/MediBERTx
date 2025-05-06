from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("../models/sentiment-bert")
model = BertForSequenceClassification.from_pretrained("../models/sentiment-bert")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)