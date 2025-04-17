import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# Dynamically find the parent directory of the current file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py')))
import config


class Head(nn.Module):
    """Singular head for self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(config.num_embd, head_size, bias=False)
        self.query = nn.Linear(config.num_embd, head_size, bias=False)
        self.value = nn.Linear(config.num_embd, head_size, bias=False)

        # add dropout layer for regularisation
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        # dont think I need this line.
        B,T,C = x.shape

        k = self.key(x)         # (B,T,head_size)
        q = self.query(x)       # (B,T,head_size)

        # compute attention scores ('affinities' between tokens) using 'scaled attention'
        # (B, T, head_size) @ (B, head_size, T) -> (B,T,T) | Note: head_size == attention_dimension
        weights = q @ torch.transpose(k, dim0=1, dim1=2) * k.shape[-1]**-0.5

        if attention_mask is not None:
            # mask: (B,T) -> (B, 1, T)
            attention_mask = attention_mask.unsqueeze(1)
            # mask out the padding tokens (real tokens dont interact with padding tokens)
            weights = weights.masked_fill(attention_mask == 0, float('-inf'))

        # apply softmax transformation for each row.
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        # randomly prevents some of the nodes / tokens from communicating / interacting
        weights = self.dropout(weights)

        # weighted aggregation on the values, v
        v = self.value(x)
        # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """Class for multi-headed self attention"""
    # different heads are responsible for specialising in learning something different, even if # parameters is the same as one vs many heads

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size , config.num_embd)
        # dropout layer for regularisation
        self.dropout = nn.Dropout(config.dropout) 

    def forward(self, x, attention_mask=None):
        # concatenate on the channel dimension
        # output from self-attention
        out_self_attention = torch.cat([head(x, attention_mask=attention_mask) for head in self.heads], dim=-1)
        # apply projection (linear-transformation) to the self-attention output, project back to the residual pathway
        out = self.dropout(self.projection(out_self_attention))
        return out
    
class FeedForwardNetwork(nn.Module):
    """A simple feed-forward network, with ReLU activation"""

    def __init__(self, num_embd):
        super().__init__()
        self.network = nn.Sequential(
            # apply linear layer on the per-token level, all tokens do this independently
            nn.Linear(num_embd, 4 * num_embd),
            nn.ReLU(),
            # projection (linear transformation) layer, back into the residual pathway
            nn.Linear(4 * num_embd, num_embd),
            # regularisation by adding dropout layers
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.network(x)
    
class Block(nn.Module):
    """Transformer block : communication (attention) followed by computation"""

    def __init__(self, num_embd, num_heads):
        # num_embd : embedding dimension
        # num_heads : the number of heads for multi-headed self-attention
        super().__init__()
        # Split the embedding size evenly across attention heads to ensure parallel processing and efficient computation.
        head_size = num_embd // num_heads
        # multi-headed self-attention mechanism (communication)
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        # computation from feed-forward network on all tokens independently
        self.feedforward = FeedForwardNetwork(num_embd)
        # add layer-normalisation, for 0-mean and unit(1)-variance across the rows.
        self.layer_norm1 = nn.LayerNorm(num_embd) 
        self.layer_norm2 = nn.LayerNorm(num_embd)

    def forward(self, x, attention_mask=None):
        # add x to output of communication and computation (residual/skip connections) and
        # apply layer normalisation before the transformations (pre-norm formulation)
        x = x + self.self_attention(self.layer_norm1(x), attention_mask=attention_mask)
        x = x + self.feedforward(self.layer_norm2(x))
        return x
    