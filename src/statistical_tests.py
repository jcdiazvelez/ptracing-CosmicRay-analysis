#!/usr/local/bin/python

import numpy as np
import healpy as hp
from scipy.stats import distributions
from matplotlib import pyplot as plt
from data_methods import rotate_map


# Kolmogorov-Smirnov test for two weighted distributions
def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    combined = np.concatenate([data1, data2])
    c_wei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    c_wei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = c_wei1[np.searchsorted(data1, combined, side='right')]
    cdf2we = c_wei2[np.searchsorted(data2, combined, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the largest of (n1, n2)
        expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)
    return d, prob


# Get list of events and weights for a given pixel
def get_pixel_distribution(pixel):
    energies = [particle[0] for particle in pixel]
    weights = [particle[1] for particle in pixel]
    return np.array([energies, weights])


def get_sky_distribution(pixel_list):
    energies = []
    weights = []
    for pixel in pixel_list:
        distribution = get_pixel_distribution(pixel)
        energies += distribution[0].tolist()
        weights += distribution[1].tolist()
    return np.array([energies, weights])


# Read in data file
filename = "toy.npz"
path = "../data/" + filename

# Physical cosmic ray distribution goes with E^(-2.7), ours goes with E^(-1)
g = 2.7

# Physical constants for scaling momentum
c = 299792458
e = 1.60217663 * 10 ** (-19)
m_p = 1.67262192 * 10 ** (-27)

# Matrix for converting from heliospheric to equatorial coordinates
equatorial_matrix = np.matrix([[-0.202372670869508942, 0.971639226673224665, 0.122321361599999998],
                               [-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],
                               [-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]])

map_matrix = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

data = np.load(path, allow_pickle=True)

# We need particle data
particles = 0

# Read in data from file
for key in data:
    if key == 'particles':
        particles = data[key]

npix = len(particles)

sky_distribution = get_sky_distribution(particles)

p_values = np.zeros(npix)
for i in range(npix):
    print(i)
    pixel_distribution = get_pixel_distribution(particles[i])
    p_values[i] = ks_weighted(pixel_distribution[0], sky_distribution[0],
                              pixel_distribution[1], sky_distribution[1])[1]

p_values = rotate_map(p_values, equatorial_matrix, map_matrix)

plt.set_cmap('bone')

p_values_smoothed = hp.smoothing(p_values, fwhm=np.radians(3.))

hp.visufunc.mollview(p_values,
                     title="P Values",
                     unit="p")
#    min=0,
#    max=0.3)
hp.graticule()

plt.savefig('P-Values')
plt.clf()
