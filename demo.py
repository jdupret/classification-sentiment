import gradio as gr
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Charger le modèle
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.load_state_dict(torch.load("model/model.pt", map_location=torch.device("cpu")))
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encodage des sentiments
df = pd.read_csv("data/train-3 - train-3.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df["sentiment"])

# Fonction de prédiction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Interface Gradio
demo = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text", title="Classification des sentiments")
demo.launch()