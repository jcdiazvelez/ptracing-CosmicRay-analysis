import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from data_methods import weight_powerlaw, rotate_map


def create_fig(sky_map, title, path, energy):
    hp.visufunc.mollview(sky_map)
    hp.graticule(coord='E')
    plt.set_cmap('coolwarm')
    plt.title(title + ' for E = ' + str(energy) + ' TeV')
    plt.savefig(path + title + '-' + str(energy) + 'TeV')


# Read in data file
filename = "new.npz"
path = "../data/" + filename

# Path for outputting figures
fig_path = '../figs/figs_helio/'

# Physical cosmic ray distribution goes with E^(-2.7), ours goes with E^(-1)
g = -2.7
power = -1

# Physical constants for scaling momentum
c = 299792458
e = 1.60217663 * 10 ** (-19)
m_p = 1.67262192 * 10 ** (-27)

# Matrix for converting from heliospheric to equatorial coordinates
equatorial_matrix = np.matrix([[-0.202372670869508942, 0.971639226673224665, 0.122321361599999998],
                               [-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],
                               [-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]])

map_matrix = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

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
    bin_midpoint = 10 ** ((np.log10(bin_sizes[i]) + np.log10(bin_sizes[i+1])) / 2.0)
    bin_weights.append(weight_powerlaw(bin_midpoint, bin_sizes[0], bin_sizes[-1], g, power))
    initial_maps[i] = rotate_map(initial_maps[i], equatorial_matrix, map_matrix)
    final_maps[i] = rotate_map(final_maps[i], equatorial_matrix, map_matrix)
    reweighed_maps[i] = rotate_map(reweighed_maps[i], equatorial_matrix, map_matrix)


# Initialise combined maps
initial = np.zeros(np.shape(initial_maps[0]))
final = np.zeros(np.shape(final_maps[0]))
reweighed = np.zeros(np.shape(reweighed_maps[0]))

map_weights = []
bin_midpoints = []

# Populate combined maps
for i in range(1, len(initial_maps) - 1):
    initial += initial_maps[i]
    final += final_maps[i]
    reweighed += reweighed_maps[i]
    map_weights.append(sum(reweighed_maps[i]))

    bin_midpoint = 10 ** ((np.log10(bin_sizes[i]) + np.log10(bin_sizes[i+1])) / 2.0)
    bin_energy = bin_midpoint * m_p * c * c / (e * 10 ** 12)
    bin_midpoints.append(bin_energy)
    bin_energy = int(bin_energy)

    # Create figures
    create_fig(hp.remove_monopole(initial_maps[i]), 'Initial Momenta', fig_path, bin_energy)
    create_fig(hp.remove_monopole(final_maps[i]), 'Final Momenta', fig_path, bin_energy)
    create_fig(hp.remove_monopole(reweighed_maps[i]), 'Reweighed Momenta', fig_path, bin_energy)

# Create figures
hp.visufunc.mollview(initial)
hp.graticule(coord='E')
plt.title('Initial Momenta (Combined)')
plt.savefig('../figs/figs_helio/initial')

hp.visufunc.mollview(final)
hp.graticule(coord='E')
plt.title('Final Momenta (Combined)')
plt.savefig('../figs/figs_helio/final')

hp.visufunc.mollview(reweighed)
hp.graticule(coord='E')
plt.title('Reweighed Initial Momenta (Combined)')
plt.savefig('../figs/figs_helio/reweighed')

plt.clf()
plt.loglog(bin_midpoints, map_weights, label='Weighted Simulation Distribution')
plt.loglog(bin_midpoints, np.power(bin_midpoints, g+1), label='Physically Observed')
plt.title('Check of particle weighting scheme')
plt.legend()
plt.savefig('../figs/figs_helio/weighting-check')



