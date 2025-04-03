from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

this_dir = Path(__file__).parent.resolve()
out_dir = this_dir / "output_mmd"
out_dir.mkdir(exist_ok=True)

def gaussian_kernel_matrix(X, Y, bandwidth):
    """Compute the gaussian kernel matrix between two sets of events

    Args:
        X (ndarray): Array containing the features of the first set of events (shape: [features, events])
        Y (ndarray): Array containing the features of the second set of events (shape: [features, events])
        bandwidth (List): List containing the bandwidth for each feature 

    Returns:
        ndarray: The Gaussian kernel matrix
    """
    X_norm = np.sum((X / bandwidth[:, None]) ** 2, axis=0)
    Y_norm = np.sum((Y / bandwidth[:, None]) ** 2, axis=0)
    K = np.exp(-0.5 * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot((X / bandwidth[:, None]).T, (Y / bandwidth[:, None]))))
    return K

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

    K_X = np.sum(gaussian_kernel_matrix(X, X, bandwidth)) / (n * n)
    K_Y = np.sum(gaussian_kernel_matrix(Y, Y, bandwidth)) / (m * m)
    K_XY = np.sum(gaussian_kernel_matrix(X, Y, bandwidth)) / (n * m)

    return K_X + K_Y - 2 * K_XY

def plot_distributions(X, Y, bins, extra_label=""):
    """Plot the distributions of the features

    Args:
        X (DataFrame): Dataframe (shape = (features, events)) containing the events of the first distribution
        Y (DataFrame): Dataframe (shape = (features, events)) containing the events of the second distribution
    """
    plt.figure(figsize=(10, 6))
    for feat in X.T.columns:
        plt.hist(X.T[feat], bins=bins, alpha=0.5, label=f'X {feat}', density=True)
        plt.hist(Y.T[feat], bins=bins, alpha=0.5, label=f'Y {feat}', density=True)
        plt.title(f'Distribution of {feat}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        if extra_label:
            extra_label = f"_{extra_label}"
        plt.savefig(out_dir / f"feature_{feat}_distribution{extra_label}.pdf")
        plt.close()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)  # For reproducibility
    NB_EVENTS_1 = 500
    NB_EVENTS_2 = 700
    BINS = 12
    EXTRA_LABEL = "test"

    # Define the number of features and their parameters
    NUM_FEATURES = 10
    FEATURE_PARAMS = [
        {'loc': np.random.uniform(-1, 1), 'scale': np.random.uniform(0, 2)} 
        for _ in range(NUM_FEATURES)
    ]
    FEATURE_PARAMS_Y = [
        {'loc': np.random.uniform(-1, 1), 'scale': np.random.uniform(0, 2)} 
        for _ in range(NUM_FEATURES)
    ]

    # First distribution
    X = pd.DataFrame({
        f'feature{i+1}': np.random.normal(
            loc=FEATURE_PARAMS[i]['loc'], 
            scale=FEATURE_PARAMS[i]['scale'], 
            size=NB_EVENTS_1
        ) for i in range(NUM_FEATURES)
    }).T

    # Second distribution
    Y = pd.DataFrame({
        f'feature{i+1}': np.random.normal(
            loc=FEATURE_PARAMS_Y[i]['loc'], 
            scale=FEATURE_PARAMS_Y[i]['scale'], 
            size=NB_EVENTS_2
        ) for i in range(NUM_FEATURES)
    }).T
    bandwidth = [1.06 * np.std(X.iloc[i, :]) * (len(X.iloc[i, :]) ** (-1 / 5)) for i in range(NUM_FEATURES)]  # Rule-of-thumb bandwidth estimate (Silverman's rule)
    
    mmd_value = mmd(X.values, Y.values, np.array(bandwidth))
    print(f"MMD: {mmd_value}")

    # Plot the distributions
    plot_distributions(X, Y, BINS, EXTRA_LABEL)