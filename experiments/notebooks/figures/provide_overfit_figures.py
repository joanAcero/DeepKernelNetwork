import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Dataset generators (2D versions for visualization)
# -----------------------------

def ds_radial(n=1500, seed=0):
    rng = np.random.default_rng(seed)

    Xi = rng.standard_normal((n, 4))

    r = np.sqrt((Xi[:, 0] ** 2 + Xi[:, 1] ** 2))
    y = (((r > 1.0).astype(int) + (Xi[:, 2] * Xi[:, 3] > 0).astype(int)) % 2)

    # return ONLY informative 2D projection
    return Xi[:, 0], Xi[:, 1], y


def ds_spiral(n=1500, seed=0, turns=2.5, noise=0.18):
    rng = np.random.default_rng(seed)
    nh = n // 2

    def arm(sign, m):
        t = np.sqrt(rng.random(m)) * turns * 2 * np.pi
        rr = t / (2 * np.pi * turns)
        x = sign * rr * np.cos(t) + rng.normal(0, noise, m)
        z = sign * rr * np.sin(t) + rng.normal(0, noise, m)
        return np.stack([x, z], 1)

    X1 = arm(1, nh)
    X2 = arm(-1, n - nh)

    X = np.vstack([X1, X2])
    y = np.array([0] * nh + [1] * (n - nh))

    order = rng.permutation(n)
    X, y = X[order], y[order]

    return X[:, 0], X[:, 1], y


def ds_checker(n=1500, seed=0, freq=3):
    rng = np.random.default_rng(seed)

    Xi = rng.uniform(-1, 1, (n, 2))

    y = (
        np.floor((Xi[:, 0] + 1) * freq).astype(int)
        + np.floor((Xi[:, 1] + 1) * freq).astype(int)
    ) % 2

    return Xi[:, 0], Xi[:, 1], y


# -----------------------------
# Plotting function
# -----------------------------

def plot_datasets():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = [
        ("Radial", ds_radial),
        ("Spiral", ds_spiral),
        ("Checker", ds_checker),
    ]

    for ax, (name, fn) in zip(axes, datasets):
        x, y, label = fn(seed=0)

        ax.scatter(x[label == 0], y[label == 0], s=6, alpha=0.6)
        ax.scatter(x[label == 1], y[label == 1], s=6, alpha=0.6)

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("datasets_overview.pdf", bbox_inches="tight")
    plt.show()


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    plot_datasets()