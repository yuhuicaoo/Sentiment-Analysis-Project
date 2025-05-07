from transformers import AutoTokenizer
from datasets import load_dataset
from data_prreprocessing import load_data, get_max_len, get_data_loader
from train import train_model, evaluate_model
import torch
import torch.nn as nn
import config
from model.sentiment_model import SentimentModel

def main():
    # load sentiment analysis dataset.
    ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

    # intialise tokeniser and data augmenter.
    tokeniser = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Get datasets
    train_ds, val_ds, test_ds = load_data(ds)

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

    test_encodings = tokeniser(
        test_ds["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Get dataloaders
    train_dataloader = get_data_loader(train_encodings, train_ds['label'], batch_size=config.batch_size, shuffle=True)
    val_dataloader = get_data_loader(val_encodings, val_ds['label'], batch_size=config.batch_size, shuffle=False)
    test_dataloader = get_data_loader(test_encodings, test_ds['label'], batch_size=config.batch_size, shuffle=False)

    model = SentimentModel(vocab_size=tokeniser.vocab_size, num_classes=3).to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay=1e-2)

    train_model(train_dataloader, val_dataloader, optimiser, model, loss_fn, config.device, epochs=10, patience=3)
    evaluate_model(model, test_dataloader, config.device)

if __name__ == "__main__":
    main()