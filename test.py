import torch
import opt_einsum


torch.backends.opt_einsum.strategy = "auto"
print("Success!pyth")