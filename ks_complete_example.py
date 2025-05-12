import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, norm

# Example: Load your data (replace with actual loading)
# Assuming data_df and mc_df are pandas DataFrames with the same columns
# e.g. data_df = pd.read_csv('data.csv')
#       mc_df = pd.read_csv('mc.csv')

ALPHA = 0.05  # significance level

# Simulated example (for demonstration)
np.random.seed(42)
data_df = pd.DataFrame(
    {
        "pt": np.random.exponential(1.0, 10000),
        "eta": np.random.normal(0, 1, 10000),
        "phi": np.random.uniform(-np.pi, np.pi, 10000),
    }
)

mc_df = pd.DataFrame(
    {
        "pt": np.random.exponential(1.0, 10000),
        "eta": np.random.normal(0, 1.1, 10000),
        "phi": np.random.uniform(-np.pi, np.pi, 10000),
    }
)
# Perform KS test with bootstrapping on each feature
ks_results = {}
n_bootstrap = 5000  # Number of bootstrap samples

for col in data_df.columns:
    stat, pval = ks_2samp(data_df[col], mc_df[col])
    bootstrap_stats = []

    # Bootstrapping
    for _ in range(n_bootstrap):
        data_sample = data_df[col].sample(frac=1, replace=True).values
        mc_sample = mc_df[col].sample(frac=1, replace=True).values
        boot_stat, _ = ks_2samp(data_sample, mc_sample)
        bootstrap_stats.append(boot_stat)

    # Perform Gaussian fit on bootstrap statistics
    mean_bootstrap, sigma_bootstrap = norm.fit(bootstrap_stats)

    # Plot Gaussian fits
    plt.figure(figsize=(12, 5))

    # Plot KS statistic bootstrap distribution
    sns.histplot(
        bootstrap_stats,
        stat="density",
        bins=30,
        color="blue",
        label="Bootstrap KS Stats",
    )
    x = np.linspace(min(bootstrap_stats), max(bootstrap_stats), 100)
    plt.plot(
        x,
        norm.pdf(x, mean_bootstrap, sigma_bootstrap),
        "r-",
        label=f"Gaussian Fit\nμ={mean_bootstrap:.3f}, σ={sigma_bootstrap:.3f}",
    )
    plt.title(f"Bootstrap KS Statistic Distribution: {col}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{col}_bootstrap_fits.pdf")

    # Store results
    ks_results[col] = {
        "KS Statistic": stat,
        "p-value": pval,
        "Bootstrap Mean KS distance": mean_bootstrap,
        "Bootstrap Sigma KS distance": sigma_bootstrap,
    }
# Compute the rejection threshold
D_LIM = np.sqrt(
    -np.log(ALPHA / 2) * (1 + len(data_df) / len(mc_df)) / (2 * len(data_df))
)
print(f"Rejection threshold (D_LIM) at {ALPHA * 100}%: {D_LIM:.4f}")
# Check if the KS statistic is greater than the threshold
for col in ks_results:
    ks_results[col][f"No Data/MC agreement at {ALPHA * 100}% rejection level"] = (
        ks_results[col]["Bootstrap Mean KS distance"] > D_LIM
    )
# Display results
results_df = pd.DataFrame(ks_results).T
print(results_df)
# Save results to a JSON file
# output_file = "ks_results.json"
# with open(output_file, "w") as f:
#     json.dump(ks_results, f, indent=4)
# print(f"KS test results saved to {output_file}")

# Optional: visualize distributions and cumulative distributions
for col in data_df.columns:
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(
        data_df[col], label="Data", stat="density", kde=True, color="black", bins=30
    )
    sns.histplot(
        mc_df[col],
        label="MC",
        stat="density",
        kde=True,
        color="red",
        bins=30,
        alpha=0.6,
    )
    plt.title(f"Histogram: {col}")
    plt.legend()

    # CDF
    plt.subplot(1, 2, 2)
    data_sorted = np.sort(data_df[col])
    mc_sorted = np.sort(mc_df[col])
    plt.step(
        data_sorted,
        np.arange(len(data_sorted)) / len(data_sorted),
        label="Data",
        color="black",
    )
    plt.step(
        mc_sorted, np.arange(len(mc_sorted)) / len(mc_sorted), label="MC", color="red"
    )
    plt.title(f"Empirical CDF: {col}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{col}_ks_test.pdf")
