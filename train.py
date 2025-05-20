import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from tqdm import tqdm

# Chargement des données
df = pd.read_csv("data/train-3 - train-3.csv")
texts = df["text"].tolist()
sentiments = df["sentiment"].tolist()

# Encodage des labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(sentiments)

# Dataset personnalisé
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Préparation
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_ds = SentimentDataset(train_texts, train_labels, tokenizer)
val_ds = SentimentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Initialisation du modèle
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Entraînement avec affichage
for epoch in range(2):
    print(f"🔁 Début époque {epoch+1}/2")
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Époque {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Fin époque {epoch+1} - Loss moyenne : {total_loss/len(train_loader):.4f}")

# Sauvegarde
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/model.pt")
print("✅ Modèle entraîné et sauvegardé dans 'model/model.pt'")