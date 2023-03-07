#!/usr/local/bin/python

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

from data_methods import create_bin_sizes, bin_particles, create_reweighed_sky_maps, \
    create_time_maps, get_reweighed_particles, rotate_map, rotate_map_sim
from statistical_methods import perform_kolmogorov_smirnov

# Read in particles
particles_filename = '../data/nside=32.npz'
particles_file = np.load(particles_filename)
particle_array = particles_file['particles']

# Set up parameters for tests
nside = 32
npix = hp.nside2npix(nside)

# Create a sky map for each bin, for weighing by energy
initial_map = np.zeros(npix)

# Populate initial and final maps
for item in particle_array:
    initial_pixel = int(item[0])
    initial_map[initial_pixel] += 1

plt.set_cmap('coolwarm')
hp.visufunc.mollview(initial_map * np.sum(initial_map) / npix,
                     # title=f'Relative intensity at edge of simulation for ' + "{0:.3g}".format(lower_limit) +
                     # title="{0:.3g}".format(lower_limit) +
                     #      ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                     title='Initial distribution at Earth',
                     unit="Relative Flux",
                     min=0.999,
                     max=1.001)
hp.graticule()
plt.savefig('../figs/initial_back_propagation')
