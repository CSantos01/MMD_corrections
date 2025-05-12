import numpy as np
from scipy.stats import ks_2samp

# Example data: two samples
sample1 = np.random.normal(loc=0.0, scale=1.0, size=500)
sample2 = np.random.normal(loc=0.5, scale=1.0, size=500)

# Perform the two-sample Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(sample1, sample2)

# Display the results
print("KS Statistic:", ks_statistic)
print(f"P-value: {p_value:.2f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: the samples come from different distributions.")
else:
    print("Fail to reject the null hypothesis: no evidence the distributions differ.")
