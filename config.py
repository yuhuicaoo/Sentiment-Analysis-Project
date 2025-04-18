import torch

# hyperparamters - change values around for testing.
batch_size = 64
block_size = 256
learning_rate = 3e-4
num_embd = 384
num_heads = 6
num_layers = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------