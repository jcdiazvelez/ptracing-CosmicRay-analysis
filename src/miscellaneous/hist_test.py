import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

# Generate chi-squared distributed data
df = 10  # degrees of freedom
chi2_data = np.random.chisquare(df, size=1000)

# Calculate chi-squared values per degree of freedom
chi2_ndf = chi2_data / df

# Create histogram of chi-squared/ndf values
plt.hist(chi2_ndf, bins=30, color='blue', alpha=0.5, density=True, histtype='step', linewidth=1.5)

# Plot the chi-squared distribution function for the corresponding degrees of freedom
x = np.linspace(0, 5, 1000)  # Adjust range as needed
y = chi2.pdf(x * df, df)  # Scaling by df to get chi-squared distribution
plt.plot(x, y, 'r-', lw=2, label=f'Chi-squared ({df} df)')

# Add labels and title
plt.xlabel(r'$\chi^2$ / ndf')
plt.ylabel('Frequency')
plt.title('Histogram of $\chi^2$/ndf with Chi-squared Distribution')
plt.legend()

# Set x-axis to logarithmic scale
plt.xscale('log')

# Show plot
plt.grid(True)
plt.show()

