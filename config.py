import torch

# hyperparamters - change values around for testing.
batch_size = 32
block_size = 128
learning_rate = 3e-4
num_embd = 192
num_heads = 6
num_layers = 6
dropout = 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------