import torch 
import torch.nn as nn
import os

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

def train_model(train_loader, val_loader, optimiser, model, loss_fn, device, epochs=5):
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

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
        print(f'LOSS train {avg_loss:.4f} validation {avg_vloss:.4f}')

    # save model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    models_count = len(os.listdir(model_dir))

    model_name = os.path.join(model_dir, f'model{models_count + 1}.pth')
    torch.save(model.state_dict(), model_name)
                