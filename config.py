import torch

# hyperparamters - change values around for testing.
batch_size = 64
block_size = 64
learning_rate = 3e-4
num_embd = 96
num_heads = 4
num_layers = 3
dropout = 0.4
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------