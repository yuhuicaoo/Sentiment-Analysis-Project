from transformers import AutoTokenizer
from datasets import load_dataset
import spacy
from data_prreprocessing import load_data, get_max_len, get_data_loader
from train import train_model
import torch
import torch.nn as nn
import config
from model.sentiment_model import SentimentModel


if __name__ == "__main__":
    # load sentiment analysis dataset.
    ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

    # intialise tokeniser.
    tokeniser = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # intialise stop_words from spacy
    nlp = spacy.load("en_core_web_sm")
    spacy_stopwords = nlp.Defaults.stop_words

    train_ds, val_ds, test_ds = load_data(ds, spacy_stopwords)

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
    train_dataloader = get_data_loader(train_encodings, train_ds['label'], batch_size=8, shuffle=True)
    val_dataloader = get_data_loader(val_encodings, val_ds['label'], batch_size=8, shuffle=False)

    # for i, data in enumerate(train_dataloader):
    #     if i > 5:
    #         break
    #     print(data)
    #     print(data['attention_mask'].shape)
    #     print(data['input_ids'].shape)
    #     print(data['label'].shape)

    model = SentimentModel(vocab_size=tokeniser.vocab_size, num_classes=3).to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    train_model(train_dataloader, val_dataloader, optimiser, model, loss_fn, config.device, epochs=5)