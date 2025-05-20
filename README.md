# Classification de Sentiments avec BERT et Gradio

## Auteur

Projet réalisé par [@jdupret](https://github.com/jdupret)

Ce projet est une application de classification de texte utilisant un modèle BERT fine-tuné. Il permet de prédire le **sentiment** (positive, negative, neutral) associé à un texte, avec une interface utilisateur via **Gradio**.

---

## Structure du projet

```
classification-sentiment/
├── data/
│   └── train-3 - train-3.csv     # Base de données (text, sentiment)
├── model/
│   └── model.pt                  # Modèle fine-tuné
├── train.py                      # Construction et entraînement du modèle
├── demo.py                       # Interface utilisateur avec Gradio
├── requirements.txt              # Dépendances Python
└── README.md                     # Rapport

---
## Entraînement du modèle

Le modèle utilise :

- `BertForSequenceClassification` (base : `bert-base-uncased`)
- Tokenization avec `BertTokenizer`
- Optimiseur `AdamW`
- Entraînement sur 2 époques avec affichage de la loss

Le modèle est sauvegardé dans `model/model.pt`.

```

### 1. Depot du projet avec Git LFS

```bash
git clone https://github.com/jdupret/classification-sentiment.git
cd classification-sentiment
git lfs install
git lfs pull
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’interface Gradio

```bash
python demo.py
```
Adresse : [http://localhost:7860](http://localhost:7860)

---

## Exemple d’utilisation

Texte  : I love this product! Highly recommended.
Sentiment : positive

```

---

## Remarques

- Le fichier `model.pt` est **stocké avec Git LFS** (car > 100 Mo).
- Pour réentraîner le modèle ou le mettre à jour, exécute `train.py`.

---
