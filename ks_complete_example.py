import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plothist as plh
import seaborn as sns
from scipy.stats import ks_2samp, norm

plh.set_style("default")

# Example: Load your data (replace with actual loading)
# Assuming data_df and mc_df are pandas DataFrames with the same columns
# e.g. data_df = pd.read_csv('data.csv')
#       mc_df = pd.read_csv('mc.csv')

ALPHA = 0.05  # significance level

# Simulated example (for demonstration)
np.random.seed(42)
DATASET = np.random.exponential(1.0, 20000)
RANDOM_DATASET = np.random.choice(DATASET, size=len(DATASET) // 2, replace=False)
data_dict = {
    "Same distribution and same dataset": RANDOM_DATASET,
    "Same distribution but different dataset": np.random.exponential(1.0, 10000),
    "Not same distribution": np.random.normal(0.5, 1, 10000),
    "Subsample": DATASET,
    "With bkg": np.random.normal(0, 1, 10000),
    "Small dataset not same": np.random.normal(0, 1, 500),
    "Small dataset same": np.random.normal(0, 1, 300),
    "1000": np.random.normal(0, 1, 1000),
}

mc_dict = {
    "Same distribution and same dataset": np.setdiff1d(DATASET, RANDOM_DATASET),
    "Same distribution but different dataset": np.random.exponential(1.0, 10000),
    "Not same distribution": np.random.normal(0, 1, 10000),
    "Subsample": RANDOM_DATASET,
    "With bkg": np.concatenate(
        [np.random.normal(0, 1, 9500), np.random.normal(1, 0.2, 500)]
    ),  # Adding a small peaking background in the tail
    "Small dataset not same": np.random.normal(0.2, 1, 500),
    "Small dataset same": np.random.normal(0, 1, 300),
    "1000": np.random.normal(0, 5.85, 1000),
}

# Perform KS test with bootstrapping on each feature
ks_results = {}
n_bootstrap = 5000  # Number of bootstrap samples

for col in data_dict.keys():
    stat, pval = ks_2samp(data_dict[col], mc_dict[col])
    bootstrap_stats = []

    # Bootstrapping
    for _ in range(n_bootstrap):
        data_sample = np.random.choice(
            data_dict[col], size=len(data_dict[col]), replace=True
        )
        mc_sample = np.random.choice(mc_dict[col], size=len(mc_dict[col]), replace=True)
        boot_stat, _ = ks_2samp(data_sample, mc_sample)
        bootstrap_stats.append(boot_stat)

    # Perform Gaussian fit on bootstrap statistics
    # mean_bootstrap, sigma_bootstrap = norm.fit(bootstrap_stats)
    mean_bootstrap = np.mean(bootstrap_stats)
    sigma_bootstrap = np.std(bootstrap_stats)

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
        -np.log(ALPHA / 2)
        * (1 + len(data_dict[col]) / len(mc_dict[col]))
        / (2 * len(data_dict[col]))
    )
    print(f"Rejection threshold (D_LIM) at {ALPHA * 100}%: {D_LIM:.4f}")
    # Check if the KS statistic is greater than the threshold
    ks_results[col][f"No Data/MC agreement at {ALPHA * 100}% rejection level"] = (
        ks_results[col]["KS Statistic"] > D_LIM
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
for col in data_dict.keys():
    plt.figure(figsize=(12, 5))

    # Determine common bin edges
    combined_data = np.concatenate([data_dict[col], mc_dict[col]])
    bins = np.histogram_bin_edges(combined_data, bins=30)

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(
        data_dict[col], label="Data", stat="density", kde=True, color="black", bins=bins
    )
    sns.histplot(
        mc_dict[col],
        label="MC",
        stat="density",
        kde=True,
        color="red",
        bins=bins,
        alpha=0.6,
    )
    plt.title(f"Histogram: {col}")
    plt.legend()

    # CDF
    plt.subplot(1, 2, 2)
    data_sorted = np.sort(data_dict[col])
    mc_sorted = np.sort(mc_dict[col])
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
