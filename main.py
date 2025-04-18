from transformers import AutoTokenizer
from datasets import load_dataset
import spacy
from data_prreprocessing import load_data, get_max_len, get_data_loader
from train import train_model
import torch
import torch.nn as nn
import config
from model.sentiment_model import SentimentModel

def main():
    # load sentiment analysis dataset.
    ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

    # intialise tokeniser.
    tokeniser = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # intialise stop_words from spacy
    nlp = spacy.load("en_core_web_sm")
    spacy_stopwords = nlp.Defaults.stop_words

    # remove negation words from stop_words (might help with negative sentiment)
    negation_words = {
    'not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 
    'neither', 'nor', 'cannot', 'won\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 
    'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'hasn\'t', 'haven\'t', 
    'hadn\'t', 'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'mightn\'t', 
    'mustn\'t', 'isn\'t', 'ain\'t'
    }

    stop_words = spacy_stopwords - negation_words

    train_ds, val_ds, test_ds = load_data(ds, stop_words)

    # get max length for tokeniser input.
    max_len = get_max_len(train_ds, tokeniser)

    train_encodings = tokeniser(
        train_ds["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    val_encodings = tokeniser(
        val_ds["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Get dataloader
    train_dataloader = get_data_loader(train_encodings, train_ds['label'], batch_size=config.batch_size, shuffle=True)
    val_dataloader = get_data_loader(val_encodings, val_ds['label'], batch_size=config.batch_size, shuffle=False)

    model = SentimentModel(vocab_size=tokeniser.vocab_size, num_classes=3).to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay=1e-2)

    train_model(train_dataloader, val_dataloader, optimiser, model, loss_fn, config.device, epochs=20, patience=3)

if __name__ == "__main__":
    main()