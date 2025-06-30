import torch 
import torch.nn as nn
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class EarlyStopper():
    def __init__(self, patience=1, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_val_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        # improvement
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            # reset counter if improvement
            self.counter = 0
        # validation loss gets worse
        elif val_loss > (self.min_val_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                # early stopping if patience exceeded
                self.early_stop = True

        return self.early_stop
        

def train_one_epoch(train_loader, optimiser, model, loss_fn, device):
    running_loss = 0

    for _, data in enumerate(train_loader):
        inputs, atn_mask, labels = data['input_ids'], data['attention_mask'], data['label']
        inputs, atn_mask, labels = inputs.to(device), atn_mask.to(device), labels.to(device)

        # zero gradients for each batch
        optimiser.zero_grad()

        # get predictions for batch
        logits = model(inputs, atn_mask)

        # compute loss and gradients
        loss = loss_fn(logits, labels)
        loss.backward()

        # adjust weights
        optimiser.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def train_model(train_loader, val_loader, optimiser, model, loss_fn, device, epochs=5, patience=1):
    early_stopper = EarlyStopper(patience=patience, delta=1e-2)
    
    print(f"Model Training Start")
    for epoch in range(epochs):
        model.train()
        avg_loss = train_one_epoch(train_loader, optimiser, model, loss_fn, device)

        # set model to evaluation mode
        model.eval()

        running_loss = 0
        # disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for _, vdata in enumerate(val_loader):
                vinputs, vatn_mask, vlabels = vdata['input_ids'], vdata['attention_mask'], vdata['label']
                vinputs, vatn_mask, vlabels = vinputs.to(device), vatn_mask.to(device), vlabels.to(device)

                vout = model(vinputs, vatn_mask)
                vloss = loss_fn(vout, vlabels)
                running_loss += vloss.item()
        
        avg_vloss = running_loss / len(val_loader)
        print(f'EPOCH {epoch + 1} | LOSS train {avg_loss:.4f} validation {avg_vloss:.4f}')

        # check for early stopping
        if early_stopper(avg_vloss):
            print(f"Early stopping triggered at Epoch {epoch + 1}")
            break
    
    print(f"Model Training Finished")

    print(f"Model Saving")
    # save model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    models_count = len(os.listdir(model_dir))

    model_name = os.path.join(model_dir, f'model{models_count + 1}.pth')
    torch.save(model.state_dict(), model_name)
    print(f"Model Saved")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data in test_loader:
            inputs, atn_mask, labels = data['input_ids'], data['attention_mask'], data['label']
            inputs, atn_mask, labels = inputs.to(device), atn_mask.to(device), labels.to(device)

            logits = model(inputs, atn_mask)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, preds = torch.max(probs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

                

