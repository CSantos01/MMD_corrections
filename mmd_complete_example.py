import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Simulate Data & MC
# --------------------------
np.random.seed(42)
n_data = 500
n_mc = 500

X_data = np.random.normal(loc=[50, 0, 91], scale=[10, 1.0, 2], size=(n_data, 3))  # Data
Y_mc = np.random.normal(
    loc=[52, 0.2, 90], scale=[10, 1.0, 2], size=(n_mc, 3)
)  # MC with shift

# Plot distributions per feature
feature_names = ["pT", "eta", "mass"]
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(6, 4))
    plt.hist(X_data[:, i], bins=30, alpha=0.5, label="Data", color="blue", density=True)
    plt.hist(Y_mc[:, i], bins=30, alpha=0.5, label="MC", color="orange", density=True)
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.title(f"Distribution of {feature_name} (Data vs MC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{feature_name}_distribution.pdf")
    plt.close()

# ----------------------------------
# Step 2: Normalize with Z-score
# ----------------------------------
scaler = StandardScaler()
X_data_norm = scaler.fit_transform(X_data)
Y_mc_norm = scaler.transform(Y_mc)


# ----------------------------------
# Step 3: Kernel + MMD Functions
# ----------------------------------
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


def median_heuristic(x, y):
    data = np.vstack([x, y])
    dists = cdist(data, data, "euclidean")
    triu = dists[np.triu_indices_from(dists, k=1)]
    return np.median(triu)


# ---------------------------------------------------
# Step 4: Permutation Test (1D)
# ---------------------------------------------------
def permutation_test_1d(x, y, sigma, num_permutations=1000):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    observed = compute_mmd(x, y, sigma)
    combined = np.vstack([x, y])
    n = x.shape[0]
    mmd_perms = []

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        mmd_perms.append(compute_mmd(x_perm, y_perm, sigma))

    mmd_perms = np.array(mmd_perms)
    p_val = np.mean(mmd_perms > observed)
    return observed, p_val


# ---------------------------------------------------
# Step 5: Run MMD + p-value per Feature
# ---------------------------------------------------
def mmd_per_feature_with_pval(X_data, Y_mc, feature_names=None, num_permutations=1000):
    d = X_data.shape[1]
    results = []
    for i in range(d):
        x_feat = X_data[:, i]
        y_feat = Y_mc[:, i]

        sigma_feat = median_heuristic(x_feat.reshape(-1, 1), y_feat.reshape(-1, 1))
        mmd_val, p_val = permutation_test_1d(
            x_feat, y_feat, sigma_feat, num_permutations
        )
        name = feature_names[i] if feature_names else f"Feature {i}"
        results.append((name, mmd_val, p_val))
    return results


# ---------------------------------------------------
# Step 6: Execute and Print
# ---------------------------------------------------
feature_names = ["pT", "eta", "mass"]
results = mmd_per_feature_with_pval(
    X_data_norm, Y_mc_norm, feature_names, num_permutations=1000
)

print("Feature-wise MMD² and p-values:")
for name, mmd, p in results:
    print(f"  {name:>5}: MMD² = {mmd:.6f}, p = {p:.4f}")

# ---------------------------------------------------
# Step 7: Plot MMD² per feature
# ---------------------------------------------------
labels = [name for name, _, _ in results]
mmd_vals = [mmd for _, mmd, _ in results]

plt.figure(figsize=(6, 4))
plt.bar(labels, mmd_vals, color="steelblue")
plt.ylabel("MMD²")
plt.title("Feature-wise MMD² (Data vs MC)")
plt.tight_layout()
plt.savefig("mmd_per_feature.pdf")
