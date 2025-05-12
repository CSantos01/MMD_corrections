import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# --- Step 1: Simulate data and MC ---
np.random.seed(42)
n_data = 500
n_mc = 800

# Simulate 3 physics features: [pT, eta, mass]
X_data = np.random.normal(loc=[50, 0, 91], scale=[10, 1.0, 2], size=(n_data, 3))  # Data
Y_mc = np.random.normal(
    loc=[51, 0.2, 90], scale=[10, 1.0, 2], size=(n_mc, 3)
)  # MC with slight offset

# --- Step 2: Normalize using Z-score ---
scaler = StandardScaler()
X_data_norm = scaler.fit_transform(X_data)
Y_mc_norm = scaler.transform(Y_mc)


# --- Step 3: Median heuristic for sigma ---
def median_heuristic(x, y):
    data = np.vstack([x, y])
    dists = cdist(data, data, "euclidean")
    triu = dists[np.triu_indices_from(dists, k=1)]
    return np.median(triu)


sigma = median_heuristic(X_data_norm, Y_mc_norm)


# --- Step 4: Compute MMD ---
def gaussian_kernel(x, y, sigma):
    dists = cdist(x, y, "sqeuclidean")
    return np.exp(-dists / (2 * sigma**2))


def compute_mmd(x, y, sigma):
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    m = x.shape[0]
    n = y.shape[0]
    return K_xx.sum() / (m * m) + K_yy.sum() / (n * n) - 2 * K_xy.sum() / (m * n)


mmd_value = compute_mmd(X_data_norm, Y_mc_norm, sigma)
print(f"MMD² = {mmd_value:.6f}")


# --- Step 5: Permutation test for significance ---
def permutation_test(x, y, sigma, num_permutations=1000):
    observed = compute_mmd(x, y, sigma)
    combined = np.vstack([x, y])
    n = x.shape[0]
    mmd_perms = []

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        mmd_perms.append(compute_mmd(x_perm, y_perm, sigma))

    p_value = np.mean(np.array(mmd_perms) > observed)
    return observed, p_value, mmd_perms


mmd_val, p_val, mmd_distribution = permutation_test(X_data_norm, Y_mc_norm, sigma)
print(f"MMD²: {mmd_val:.6f} | p-value: {p_val:.4f}")

# --- Step 6: Plot permutation distribution ---
plt.hist(mmd_distribution, bins=30, alpha=0.7, label="Permutation MMD²")
plt.axvline(
    mmd_val, color="red", linestyle="--", label=f"Observed MMD² = {mmd_val:.4f}"
)
plt.title("Permutation Test for MMD²")
plt.xlabel("MMD²")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("mmd_permutation_test.pdf")
