import torch

# hyperparamters - change values around for testing.
batch_size = 32
block_size = 128
learning_rate = 1e-4
num_embd = 128
num_heads = 4
num_layers = 4
dropout = 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------