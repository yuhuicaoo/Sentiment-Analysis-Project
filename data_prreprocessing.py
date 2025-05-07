import re
import torch
from torch.utils.data import Dataset
from numpy import percentile
from torch.utils.data import DataLoader 


def load_data(data):
    """Add docstring"""
    train_ds = data['train'].to_pandas().dropna()
    val_ds = data['validation'].to_pandas().dropna()
    test_ds = data['test'].to_pandas().dropna()

    # preprocess text for each dataset
    train_ds["text"] = train_ds["text"].apply(lambda text: preprocess_text(text))
    val_ds['text'] = val_ds['text'].apply(lambda text: preprocess_text(text))
    test_ds['text'] = test_ds['text'].apply(lambda text: preprocess_text(text))
    return train_ds, val_ds , test_ds 

def preprocess_text(text):
    """
    Add doctstring
    """
    # intialise url pattern for regex
    url_pattern = re.compile(r"https?://\S+")
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    # remove numbers from text
    text = re.sub(r"\d+", "", text)
    # remove urls / links
    text = re.sub(url_pattern, "", text)
    # lowercase text
    text = str(text).lower()
    return text

def get_max_len(dataset, tokeniser):
    """
    Docstring add later
    """
    train_lens = [len(tokeniser.encode(text, truncation=False)) for text in dataset["text"]]
    max_len = min(int(percentile(train_lens, 95)), 128)
    return max_len

# Create custom dataset class for PyTorch DataLoader
class TextDataSet(Dataset):
    def __init__(self, encodings, labels):
        # encodings is a dictionary of input_ids & attention_mask vector
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def get_data_loader(encodings, labels, batch_size=8, shuffle=True):
    dataset = TextDataSet(encodings, labels.tolist())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader