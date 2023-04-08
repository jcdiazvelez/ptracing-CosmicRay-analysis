#!/usr/local/bin/python

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.stats as stat
from argparse import ArgumentParser
from data_methods import create_bin_sizes, bin_particles, get_reweighed_particles, rotate_map
from tqdm import tqdm

# kolmogorov-p-Smirnov test for two weighted distributions
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
    diff = cdf1we - cdf2we
    abs_diff = np.abs(diff)
    d = np.max(abs_diff)
    ind = np.argmax(abs_diff)
    signed_d = diff[ind]
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = np.sqrt(m * n / (m + n))
    if alternative == 'two-sided':
        prob = stat.distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the largest of (n1, n2)
        expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)
    return signed_d, prob, cdf1we, cdf2we


# Get list of events and weights for a given pixel
def get_pixel_distribution(pixel):
    energies = [particle[0] for particle in pixel]
    weights = [particle[1] for particle in pixel]
    return np.array([energies, weights])


# Get an average distribution across the whole sky
def get_sky_distribution(pixel_list):
    energies = []
    weights = []
    for pixel in pixel_list:
        distribution = get_pixel_distribution(pixel)
        energies += distribution[0].tolist()
        weights += distribution[1].tolist()
    return np.array([energies, weights])


def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    return get_sky_distribution(pixel_list[particle_ring])


# Get the pixel distribution of particles within a declination strip of the chosen pixel
def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    theta, phi = hp.pix2ang(nside, pixel_number)
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    strip = hp.query_strip(nside, theta - num_pixels * d_theta, theta + num_pixels * d_theta)
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    strip = np.setdiff1d(strip, particle_ring)
    return get_sky_distribution(pixel_list[strip])


# Impose a fixed energy range on a distribution
def impose_energy_range(distribution, min_energy, max_energy):
    energies = distribution[0]
    weights = distribution[1]
    indices = np.where(np.logical_and(energies >= min_energy, energies <= max_energy))
    return np.array([energies[indices], weights[indices]])


# Perform Kolmorogov-Smirnov for a given set of particles, energy limits and strip width
def perform_kolmogorov_smirnov(particles, limits, width):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)

    npix = len(particles)
    nside = hp.npix2nside(npix)

    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))

    p_values = []
    for i in tqdm(range(1230,1250)):
        strip_distribution = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = ks_weighted(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1])
        p_values.append([results[2],results[3]])

    return p_values


# Parser for reading command line arguments
parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, default='../figs/',
                    help="Output directory for figure data")
parser.add_argument("-p", "--path", type=str, default="../data/",
                    help="Path to data")
parser.add_argument("-N", "--nside", type=int, default="16",
                    help="plot resolution")
parser.add_argument("-g", "--phys_index", type=float, default="-2.7",
                    help="power law index for physical cosmic ray distribution")
parser.add_argument("-P", "--model_index", type=float, default="-1.0",
                    help="power law index for modelled cosmic ray distribution")

args = parser.parse_args()
args_dict = vars(args)

# Read in particles
particles_filename = args.path + 'nside=' + str(args.nside) + '.npz'
particles_file = np.load(particles_filename, allow_pickle=True)
particle_array = particles_file['particles']

# Set up parameters for tests
nside = args.nside
npix = hp.nside2npix(nside)

ks_particles = np.array(get_reweighed_particles(particle_array, 1, args.nside,
                                                args.phys_index, args.model_index)[0])

# Perform KS test for each set of limits and widths
ks_data = perform_kolmogorov_smirnov(ks_particles, [0, 1000], 3)

for i in range(1230,1250):
    plt.loglog(ks_data[i-1230][0], label=f'Pixel = {i}')
    plt.loglog(ks_data[i-1230][1], label=f'Average')
    plt.legend()
    plt.savefig(f'../figs/pixel={i}')
    plt.close()

# Save all produced data
prefix = 'kolmogorov'
output_name = args.output + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name,
                    cumulative_data=np.array([ks_data[2],ks_data[3]]))

