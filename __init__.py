import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

this_dir = Path(__file__).parent.resolve()
out_dir = this_dir / "output_mmd"
out_dir.mkdir(exist_ok=True)


def gaussian_kernel(x, y, bandwidth):
    """Compute the gaussian kernel between two events

    Args:
        x (List): List containing the features of the first event
        y (List): List containing the features of the second event
        bandwidth (List): List containing the bandwidth for each feature

    Returns:
        Float: The value of the gaussian kernel
    """
    x = np.array(x)
    # print(f"x: {x}")
    y = np.array(y)
    # print(f"y: {y}")
    bandwidth = np.array(bandwidth)

    # Compute the Gaussian kernel using vectorized operations
    exponent = -((x - y) ** 2) / (bandwidth**2)
    kern = np.sum(exponent)
    # print(f"kern: {kern}")
    kernel = np.exp(kern)
    # print(f"kernel: {kernel}")

    return kernel


def mmd(X, Y, bandwidth):
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions

    Args:
        X (DataFrame): Dataframe (shape = (features, events)) containing the events of the first distribution
        Y (DataFrame): Dataframe (shape = (features, events)) containing the events of the second distribution
        bandwidth (List): List containing the bandwidth for each feature

    Returns:
        Float: The value of the MMD
    """
    n = X.shape[1]
    print(f"Number of events in the first dataset: {n}")
    m = Y.shape[1]
    print(f"Number of events in the second dataset: {m}")

    K_X = 0
    K_Y = 0
    K_XY = 0

    t0 = time.time()
    for i in range(n):
        for j in range(n):
            if i != j:
                K_X += gaussian_kernel(X[i], X[j], bandwidth)
    t1 = time.time()
    print(f"Kernel computation time for X: {t1 - t0:.1f} seconds")
    print(f"Value of K_X: {K_X:.2e}")

    t0 = time.time()
    for i in range(m):
        for j in range(m):
            if i != j:
                K_Y += gaussian_kernel(Y[i], Y[j], bandwidth)
    t1 = time.time()
    print(f"Kernel computation time for Y: {t1 - t0:.1f} seconds")
    print(f"Value of K_Y: {K_Y:.2e}")

    t0 = time.time()
    for i in range(n):
        for j in range(m):
            K_XY += gaussian_kernel(X[i], Y[j], bandwidth)
    t1 = time.time()
    print(f"Kernel computation time for XY: {t1 - t0:.1f} seconds")
    print(f"Value of K_XY: {K_XY:.2e}")

    return K_X / (n * (n - 1)) + K_Y / (m * (m - 1)) - 2 * K_XY / (n * m)


def plot_distributions(X, Y, bins, extra_label=""):
    """Plot the distributions of the features

    Args:
        X (DataFrame): Dataframe (shape = (features, events)) containing the events of the first distribution
        Y (DataFrame): Dataframe (shape = (features, events)) containing the events of the second distribution
    """
    plt.figure(figsize=(10, 6))
    for feat in X.T.columns:
        plt.hist(X.T[feat], bins=bins, alpha=0.5, label=f"X {feat}", density=True)
        plt.hist(Y.T[feat], bins=bins, alpha=0.5, label=f"Y {feat}", density=True)
        plt.title(f"Distribution of {feat}")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        if extra_label:
            label = f"_{extra_label}"
        plt.savefig(out_dir / f"feature_{feat}_distribution{label}.pdf")
        plt.close()


if __name__ == "__main__":
    # Example usage
    np.random.seed(420)  # For reproducibility
    NB_EVENTS_1 = 200
    NB_EVENTS_2 = 240
    BINS = 12
    EXTRA_LABEL = "__"
    NUM_FEATURES = 10

    # Define the number of features and their parameters
    FEATURE_PARAMS = [
        {"loc": np.random.uniform(-10, 10), "scale": np.random.uniform(1, 3)}
        for _ in range(NUM_FEATURES)
    ]
    FEATURE_PARAMS_Y = [
        {"loc": np.random.uniform(-10, 10), "scale": np.random.uniform(1, 2)}
        for _ in range(NUM_FEATURES)
    ]

    # First distribution
    X = pd.DataFrame(
        {
            f"feature{i + 1}": np.random.normal(
                loc=FEATURE_PARAMS[i]["loc"],
                scale=FEATURE_PARAMS[i]["scale"],
                size=NB_EVENTS_1,
            )
            for i in range(NUM_FEATURES)
        }
    ).T
    # print(f"First dataset: {X}")

    # Second distribution
    Y = pd.DataFrame(
        {
            f"feature{i + 1}": np.random.normal(
                loc=FEATURE_PARAMS_Y[i]["loc"],
                scale=FEATURE_PARAMS_Y[i]["scale"],
                size=NB_EVENTS_2,
            )
            for i in range(NUM_FEATURES)
        }
    ).T
    # print(f"Second dataset: {Y}")

    # Compute the bandwidth for each feature using Scott's rule
    bandwidth = [
        1.06
        * np.std(pd.concat([X.iloc[i, :], Y.iloc[i, :]]))
        * (len(pd.concat([X.iloc[i, :], Y.iloc[i, :]])) ** (-1 / (NUM_FEATURES + 4)))
        for i in range(NUM_FEATURES)
    ]
    print(f"Bandwidth (Scott's rule): {bandwidth}")

    t0 = time.time()
    mmd_value = mmd(X, Y, bandwidth)
    t1 = time.time()
    print(f"Total kernel computation time: {t1 - t0:.1f} seconds")
    print(f"MMD: {mmd_value:.4e}")

    # Plot the distributions
    plot_distributions(X, Y, BINS, EXTRA_LABEL)
