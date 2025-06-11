import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.random.randn(1000)

# Compute histogram without density normalization
hist_counts, bin_edges_counts = np.histogram(data, bins=10, range=(-3, 3), density=False)
# Compute histogram with density normalization
hist_density, bin_edges_density = np.histogram(data, bins=10, range=(-3, 3), density=True)

print("Histogram counts (density=False):", hist_counts)
print("Histogram density (density=True):", hist_density)
print("Bin edges:", bin_edges_counts)

# Plot histograms using Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Histogram with counts
ax1.hist(data, bins=10, range=(-3, 3), edgecolor='black')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram (density=False)')

# Histogram with density
ax2.hist(data, bins=10, range=(-3, 3), density=True, edgecolor='black')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_title('Histogram (density=True)')

plt.show()
