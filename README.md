<<<<<<< HEAD
"# classification-sentiment" 
=======
# Sentiment Classifier with PyTorch and Gradio

This project classifies text data into sentiment categories (`positive`, `neutral`, `negative`) using a fine-tuned BERT model.

## Dataset

We use the original CSV file that contains columns such as `text`, `sentiment`, etc.

## Steps

1. **Training**: Fine-tunes `BertForSequenceClassification` on the dataset.
2. **Demo**: Gradio app to test the model live.

## How to Run

```bash
pip install -r requirements.txt
python train.py
python demo.py
```

## Structure

- `train.py`: training logic with dataset class
- `demo.py`: Gradio interface
- `data/247925cb-d36a-42ea-a8df-62578103ce08.csv`: Original dataset
- `model/model.pt`: Saved trained model
>>>>>>> 5bf9191 (🎉 Initial commit: sentiment classification with BERT and Gradio)
