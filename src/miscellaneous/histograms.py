import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
data = np.random.normal(loc=0, scale=1, size=(1000, 2))  # Two datasets

# Create a figure and a set of subplots with different histogram types
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Regular histogram
axs[0, 0].hist(data[:, 0], bins=30, color='blue', alpha=0.5)
axs[0, 0].set_title('Regular Histogram')

# Step histogram
axs[0, 1].hist(data[:, 0], bins=30, color='green', alpha=0.5, histtype='step')
axs[0, 1].set_title('Step Histogram')

# Bar histogram
axs[1, 0].hist(data[:, 0], bins=30, color='orange', alpha=0.5, histtype='bar')
axs[1, 0].set_title('Bar Histogram')

# Stacked Bar Histogram
axs[1, 1].hist(data, bins=30, color=['blue', 'green'], alpha=0.7, histtype='barstacked', label=['Dataset 1', 'Dataset 2'])
axs[1, 1].set_title('Stacked Bar Histogram')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

