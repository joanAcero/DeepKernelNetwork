import matplotlib.pyplot as plt
import numpy as np

def get_plot_data(ds_fn, seed=0):
    # regenerate WITHOUT permuting dimensions
    rng = np.random.default_rng(seed)

    if ds_fn.__name__ == "ds_radial":
        Xi = rng.standard_normal((1500, 4))
        r = np.sqrt((Xi[:, :2] ** 2).sum(1))
        y = (((r > 1.0).astype(int) + (Xi[:, 2] * Xi[:, 3] > 0).astype(int)) % 2)
        return Xi[:, 0], Xi[:, 1], y

    if ds_fn.__name__ == "ds_spiral":
        nh = 750
        def arm(sign, m):
            t = np.sqrt(rng.random(m)) * 2.5 * 2 * np.pi
            rr = t / (2 * np.pi * 2.5)
            x = sign * rr * np.cos(t) + rng.normal(0, 0.18, m)
            z = sign * rr * np.sin(t) + rng.normal(0, 0.18, m)
            return np.stack([x, z], 1)

        Xi = np.vstack([arm(1, nh), arm(-1, 750)])
        y = np.array([0] * nh + [1] * 750)
        return Xi[:, 0], Xi[:, 1], y

    if ds_fn.__name__ == "ds_checker":
        Xi = rng.uniform(-1, 1, (1500, 2))
        y = ((np.floor((Xi[:, 0] + 1) * 3).astype(int) +
              np.floor((Xi[:, 1] + 1) * 3).astype(int)) % 2)
        return Xi[:, 0], Xi[:, 1], y

def plot_datasets():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    names = ["radial", "spiral", "checker"]

    for ax, name in zip(axes, names):
        fn = DATASETS[name]
        x, y, label = get_plot_data(fn, seed=0)

        ax.scatter(x[label == 0], y[label == 0], s=5, alpha=0.6, label="class 0")
        ax.scatter(x[label == 1], y[label == 1], s=5, alpha=0.6, label="class 1")

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("datasets_overview.pdf", bbox_inches="tight")
    plt.show()