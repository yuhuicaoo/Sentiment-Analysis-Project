import torch
import torch.nn as nn
from model.transformer_model import Block 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py')))
import config


class SentimentModel(nn.Module):
    """Sentiment analysis model"""
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.num_embd)
        # positional encodings for tokens 
        self.position_embedding_table = nn.Embedding(config.block_size, config.num_embd)
        self.blocks = nn.ModuleList([Block(config.num_embd, config.num_heads) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.num_embd)
        # project down to the number of classes.
        self.classifier = nn.Linear(config.num_embd, num_classes)

    def forward(self, idx, attention_mask=None):
        B,T = idx.shape

        token_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=config.device))  # (T,C)
        x = token_embd + pos_embd # (B, T, C)
        for block in self.blocks:
            x = block(x, attention_mask)  # (B,T,C)

        x = self.layer_norm(x)  # (B,T,C)

        # CLS token pooling
        cls_token_output = x[:, 0, :] # (B , C)
        print(cls_token_output)

        logits = self.classifier(cls_token_output) # (B, num_classes)
        return logits

