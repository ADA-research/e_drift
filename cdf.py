import numpy as np
import matplotlib.pyplot as plt

# Generate sample data within the range [0, 0.4]
np.random.seed(42)

# Creating more linear data distributions
reference_data = np.random.uniform(0.005, 0.4, size=500)  # Reference window data
target_data = np.random.uniform(0.0, 0.4, size=450)       # Target window data
zero_data = np.zeros(50)
target_data = np.concatenate((target_data, zero_data))

# Compute CDF
def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

reference_sorted, reference_cdf = compute_cdf(reference_data)
target_sorted, target_cdf = compute_cdf(target_data)

# Plot the CDFs
plt.figure(figsize=(8, 8))
plt.plot(reference_sorted, reference_cdf, color='pink', label='before concept drift', linewidth=3)
plt.plot(target_sorted, target_cdf, color='grey', label='after concept drift', linewidth=3)
plt.xlabel(r'$\tilde{\varepsilon}^*$ values', fontsize=20)
plt.ylabel(r'Fraction of observed $\tilde{\varepsilon}^*$ values', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig("cdf_third_approach_purple.pdf")
plt.show()


#cdf first approach met seed 0
#cdf seconda approach yellow seed 42
#cdf second approach purple seed 42
