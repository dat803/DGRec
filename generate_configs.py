import json
import itertools

gammas = [2.0, 1.0, 3.0]
sigmas = [1.0, 0.5, 2.0]
embed_sizes = [32, 64]
learning_rates = [0.05, 0.1, 0.01]
popularities = [False, True]
kernels = ["gaussian", "poly", "tanh"]
datasets = ["Beauty", "TaoBao"]

valid_kernels = {"gaussian", "tanh", "radial"}

configs = []

for gamma, sigma, embed_size, lr, popularity, kernel, dataset in itertools.product(
    gammas, sigmas, embed_sizes, learning_rates, popularities, kernels, datasets
):
    if kernel not in valid_kernels:
        continue
    if kernel == "gaussian" and sigma is None:
        continue
    if kernel in {"tanh", "radial"} and gamma is None:
        continue

    config = {
        "gamma": gamma,
        "sigma": sigma,
        "embed_size": embed_size,
        "learning_rate": lr,
        "popularity": popularity,
        "kernel": kernel,
        "dataset": dataset,
    }
    configs.append(config)

print(f"Generated {len(configs)} valid configurations.")
with open("configs.json", "w") as f:
    json.dump(configs, f, indent=2)
