#!/usr/local/bin/python

import numpy as np
import healpy as hp
from data_methods import create_bin_sizes, bin_particles, create_reweighed_sky_maps, \
    create_time_maps, get_reweighed_particles, rotate_map, rotate_map_sim
from statistical_methods import perform_kolmogorov_smirnov
from argparse import ArgumentParser

# Parser for reading command line arguments
parser = ArgumentParser()
parser.add_argument("-l", "--limits", type=str, default='./limits',
                    help="File containing limits for distribution tests")
parser.add_argument("-b", "--bins", type=str, default='./bins',
                    help="File containing binning schemes for binned tests")
parser.add_argument("-w", "--widths", type=str, default='./widths',
                    help="File containing strip widths for strip tests")
parser.add_argument("-o", "--output", type=str, default='../figs/',
                    help="Output directory for figure data")
parser.add_argument("-k", "--kolmogorov", type=bool, default=False,
                    help="Run Kolmorogov tests (time consuming)")
parser.add_argument("-p", "--path", type=str, default="../data/",
                    help="Path to data")
parser.add_argument("-N", "--nside", type=int, default="2",
                    help="plot resolution")
parser.add_argument("-g", "--phys_index", type=float, default="-2.7",
                    help="power law index for physical cosmic ray distribution")
parser.add_argument("-P", "--model_index", type=float, default="-1.0",
                    help="power law index for modelled cosmic ray distribution")

args = parser.parse_args()
args_dict = vars(args)

# Read in files
limits = np.loadtxt(args.limits)
bins = np.loadtxt(args.bins, dtype=int)
widths = np.loadtxt(args.widths, dtype=int)

# Read in particles
particles_filename = args.path + 'nside=' + str(args.nside) + '.npz'
particles_file = np.load(particles_filename)
particle_array = 0
for key in particles_file:
    if key == 'particles':
        particle_array = particles_file[key]

# Set up parameters for tests
nside = args.nside
npix = hp.nside2npix(nside)
num_binnings = len(bins)
num_limits = len(limits)
num_widths = len(widths)
max_bins = np.max(bins)

# Create initial arrays to be written to by each of the tests
reweighed_maps_initial = []
reweighed_maps_final = []
time_maps = []
kolmogorov_smirnov_distribution_maps = []
bin_limits = []

# Do binned tests for each binning
for binning in bins:
    particle_sets = get_reweighed_particles(particle_array, binning, args.nside, args.phys_index, args.model_index)
    particles = particle_sets[0]
    particles_final = particle_sets[1]

    # Sort particle data into energy bins
    bin_sizes = create_bin_sizes(particles, binning)
    bin_limits.append(np.pad(bin_sizes, (0, 1 + max_bins - len(bin_sizes))))
    binned_particles = bin_particles(particles, bin_sizes)
    binned_final = bin_particles(particles_final, bin_sizes)

    # Create reweighed initial flux maps
    print(f'Creating reweighed initial flux maps for {binning} energy bins')
    reweighed_maps_initial = [*reweighed_maps_initial, *create_reweighed_sky_maps(binned_particles)]

    # Create reweighed final flux maps
    print(f'Creating reweighed final flux maps for {binning} energy bins')
    reweighed_maps_final = [*reweighed_maps_final, *create_reweighed_sky_maps(binned_final)]

    # Create time map
    print(f'Creating time averaged maps for {binning} energy bins')
    time_maps = [*time_maps, *create_time_maps(binned_particles)]

# Perform KS test for each set of limits and widths
if args.kolmogorov:
    for limit in limits:
        for width in widths:
            print(f'Performing Kolmogorov-Smirnov test on distribution with lower = {limit[0]} TeV, upper = {limit[1]} '
                  f'TeV and width = {width}')
            ks_map = perform_kolmogorov_smirnov(particles, limit, width)
            kolmogorov_smirnov_distribution_maps.append(ks_map)

# Rotate sky maps to appropriate coordinates
reweighed_maps_initial = hp.sphtfunc.smoothing(rotate_map(reweighed_maps_initial), fwhm=0.05)
reweighed_maps_final = rotate_map_sim(reweighed_maps_final)
time_maps = rotate_map(time_maps)
kolmogorov_smirnov_distribution_maps = rotate_map(kolmogorov_smirnov_distribution_maps)

# Save all produced data
prefix = 'nside=' + str(nside)
output_name = args.output + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name,
                    flux=reweighed_maps_initial,
                    flux_final=reweighed_maps_final,
                    time=time_maps,
                    kolmogorov=kolmogorov_smirnov_distribution_maps,
                    bin_limits=bin_limits,
                    limits=limits,
                    bins=bins,
                    widths=widths)
