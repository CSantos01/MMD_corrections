import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


# --- Gaussian (RBF) kernel ---
def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian RBF kernel matrix between x and y"""
    dists = cdist(x, y, "sqeuclidean")  # pairwise squared Euclidean distances
    return np.exp(-dists / (2 * sigma**2))


# --- MMD computation ---
def compute_mmd(x, y, sigma=1.0):
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    m = x.shape[0]
    n = y.shape[0]

    mmd2 = K_xx.sum() / (m * m) + K_yy.sum() / (n * n) - 2 * K_xy.sum() / (m * n)
    return mmd2


# --- Generate sample data ---
np.random.seed(42)
x = np.random.normal(1, 1, (500, 1))  # Sample from N(0, 1)
y = np.random.laplace(
    0.5, 1, (1300, 1)
)  # Laplace distribution with mean 0.5 and scale 1

# --- Run MMD test ---
# Silverman's rule of thumb for bandwidth
sigma = 1.06 * np.std(np.vstack((x, y))) * (len(x) + len(y)) ** (-1 / 5)
mmd_value = compute_mmd(x, y, sigma=sigma)
print(f"MMD^2 = {mmd_value:.4f}")

# --- Plot distributions ---
# Plot the two distributions
plt.hist(
    x, bins=30, alpha=0.5, label="Distribution X (N(0, 1))", color="blue", density=True
)
plt.hist(
    y,
    bins=30,
    alpha=0.5,
    label="Distribution Y (N(0.5, 1))",
    color="orange",
    density=True,
)
plt.title("Distributions of X and Y")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
