from transformers import AutoTokenizer
from datasets import load_dataset
import spacy
from data_prreprocessing import TextDataSet, preprocess_text, load_data, get_max_len, get_data_loader
from numpy import percentile
from torch.utils.data import DataLoader 


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

    # Get dataloader
    train_dataloader = get_data_loader(train_encodings, train_ds['label'], batch_size=8, shuffle=True)
    a = next(iter(train_dataloader))
    print(a['input_ids'][0])
    print(tokeniser.decode(a['input_ids'][0]))
    print(a['label'][0])