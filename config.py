import torch

# hyperparamters - change values around for testing.
batch_size = 64
block_size = 64
learning_rate = 3e-4
num_embd = 128
num_heads = 4
num_layers = 4
dropout = 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------