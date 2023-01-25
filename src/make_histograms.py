import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from data_methods import weight_powerlaw, rotate_map

# Read in data file
filename = "hybrid-weight.npz"
path = "../data/" + filename

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

# Calculate midpoints of energy bins and rotate maps to plotting coordinate system
bin_midpoints = []
for i in range(len(bin_sizes) - 1):
    bin_midpoint = 10 ** ((np.log10(bin_sizes[i]) + np.log10(bin_sizes[i+1])) / 2.0)
    bin_midpoints.append(bin_midpoint * m_p * c * c / (e * 10 ** 12))

    initial_maps[i] = rotate_map(initial_maps[i], equatorial_matrix, map_matrix)
    final_maps[i] = rotate_map(final_maps[i], equatorial_matrix, map_matrix)
    reweighed_maps[i] = rotate_map(reweighed_maps[i], equatorial_matrix, map_matrix)


# Convert to histograms for each pixel

histograms = np.transpose(reweighed_maps)

for pixel in histograms:
    plt.step(np.log10(bin_midpoints), np.log10(pixel[1:] / np.power(bin_midpoints, g+1)))

plt.title('Relative weighted flux for each pixel')
plt.xlabel('log(E / TeV)')
plt.ylabel('log(Flux * E^1.7)')

plt.show()
# Perform chi squared test for each pixel
