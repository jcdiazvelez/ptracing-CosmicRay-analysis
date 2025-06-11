import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

# Generate sample data following chi-squared distribution
df = 19  # degrees of freedom
data = np.random.chisquare(df, size=1000)

# Create a histogram using dots
plt.hist(data, bins=30, color='blue', alpha=0.5, density=True, histtype='step', linestyle='dashed', linewidth=1.5)

# Plot the chi-squared distribution function
x = np.linspace(0, 40, 1000)
y = chi2.pdf(x, df)
plt.plot(x, y, 'r-', lw=2, label=f'Chi-squared ({df} df)')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Frequency')
plt.title('Histogram with Chi-squared Distribution')
plt.legend()

# Show plot
plt.grid(True)
plt.show()

