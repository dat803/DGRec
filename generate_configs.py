import json
import itertools

gammas = [2.0]
sigmas = [1.0]
learning_rates = [0.05]
popularities = ["disabled"]
kernels = ["poly", "tanh", "linear", "radial", "gaussian"]
datasets = ["TaoBao"]

configs = []

for gamma, sigma, lr, popularity, kernel, dataset in itertools.product(
    gammas, sigmas, learning_rates, popularities, kernels, datasets
):
    if kernel == "gaussian" and sigma is None:
        continue
    if kernel in {"tanh", "radial", "poly"} and gamma is None:
        continue

    config = {
        "gamma": gamma,
        "sigma": sigma,
        "learning_rate": lr,
        "popularity": popularity,
        "kernel": kernel,
        "dataset": dataset,
    }
    configs.append(config)

print(f"Generated {len(configs)} valid configurations.")
with open("configs.json", "w") as f:
    json.dump(configs, f, indent=2)
