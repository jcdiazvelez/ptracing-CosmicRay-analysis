import numpy as np
import matplotlib.pyplot as plt

def plot_data_with_buckets(data):
    # Determine bin edges
    bin_edges = np.linspace(min(data), max(data), 21)  # 20 buckets

    # Plot each data point individually
    plt.plot(data, np.zeros_like(data), 'bo', markersize=5)

    # Overlay histogram with bucket intervals
    plt.hist(data, bins=bin_edges, alpha=0.5, color='r', edgecolor='black')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Data Distribution with 20 Buckets')
    plt.grid(True)
    plt.show()

# Example usage:
data = np.random.normal(loc=0, scale=1, size=1000)
plot_data_with_buckets(data)


