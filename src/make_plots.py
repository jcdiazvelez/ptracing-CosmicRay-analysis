import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

from data_methods import apply_rotation


# Power law pdf for a given index. In our case g = -1
def powerlaw_pdf(x, x_min, x_max, g):
    x_min_g, x_max_g = x_min ** (g + 1.), x_max ** (g + 1.)
    if g == -1.0:
        return x ** g / np.log(x_max / x_min)
    else:
        return (g + 1.) / (x_max_g - x_min_g) * x ** g


# Weighting scheme for energy bins
def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)


# Read in data file
filename = "a_crossings_c_energy_.npz"
path = "../data/" + filename

# Physical cosmic ray distribution goes with E^(-2.7), ours goes with E^(-1)
g = -2.7
power = -1

data = np.load(path)

initial_maps = 0
final_maps = 0
reweighed_maps = 0
bin_sizes = 0

# Read in data from file
for key in data:
    if key == 'initial':
        initial_maps = data[key]
    elif key == 'final':
        final_maps = data[key]
    elif key == 'reweighed':
        reweighed_maps = data[key]
    elif key == 'bins':
        bin_sizes = data[key]

# Calculate weights of energy bins
bin_weights = []
for i in range(len(bin_sizes) - 1):
    bin_energy = 10 ** ((np.log10(bin_sizes[i]) + np.log10(bin_sizes[i+1])) / 2.0)
    bin_weights.append(weight_powerlaw(bin_energy, bin_sizes[0], bin_sizes[-1], g, power))
    # bin_weights.append(1)

# Initialise combined maps
initial = np.zeros(np.shape(initial_maps[0]))
final = np.zeros(np.shape(final_maps[0]))
reweighed = np.zeros(np.shape(reweighed_maps[0]))

# Populate combined maps
for i in range(len(initial_maps) - 1):
    initial += initial_maps[i] * bin_weights[i]
    final += final_maps[i] * bin_weights[i]
    reweighed += reweighed_maps[i] * bin_weights[i]

# Create figures
hp.visufunc.mollview(initial)
plt.title('Initial Momenta')
plt.savefig('../figs/initial')

hp.visufunc.mollview(final)
plt.title('Final Positions')
plt.savefig('../figs/final')

hp.visufunc.mollview(reweighed)
plt.title('Reweighed Initial Momenta')
plt.savefig('../figs/reweighed')
