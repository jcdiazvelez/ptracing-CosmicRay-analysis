#!/usr/local/bin/python

import time
import sys
import numpy as np
import os, os.path
import pickle
import glob
import healpy as hp
import matplotlib.pyplot as plt

from multiprocessing import Pool, TimeoutError

from argparse import ArgumentParser

from data_methods import create_position_maps, weight_powerlaw, cos_dipole_f

# Parser for reading command line arguments
parser = ArgumentParser()
parser.add_argument("-n", "--nfiles", type=int, default=0,
                    help="Number of files per job (for selecting subset of files)")
parser.add_argument("-p", "--path", type=str, default="../../data/h5hybrid-2/",
                    help="Path to data")
parser.add_argument("-o", "--outdir", type=str, default="../data/",
                    help="output directory")
parser.add_argument("-N", "--nside", type=int, default="30",
                    help="plot resolution")
parser.add_argument("--prefix", type=str, default="",
                    help="output directory")
parser.add_argument("-r", "--radius", type=int, default="50000",
                    help="termination radius")
parser.add_argument("-b", "--bins", type=int, default="10",
                    help="number of energy bins")
parser.add_argument("-g", "--phys_index", type=float, default="-2.7",
                    help="power law index for physical cosmic ray distribution")
parser.add_argument("-P", "--model_index", type=float, default="-1.0",
                    help="power law index for modelled cosmic ray distribution")

args = parser.parse_args()
args_dict = vars(args)

# File names for particle data
filename = "*.npz"
path = args.path + "/" + filename

print(path)

# Prepare file names for processing
files = sorted(glob.glob(path))
n_files = len(files)

print(n_files, "total files")

if args.nfiles:
    n_files = args.nfiles

print("processing", n_files, " files")

# Set healpy nside and termination radius
nside = args.nside
npix = hp.nside2npix(nside)
radius = args.radius

# Use 16 worker processes
pool = Pool(processes=16)

# Create pool input for direction data map
pool_input = []
for i in range(n_files):
    pool_input.append((files[i], nside, radius))

# Generate and flatten direction data
direction_data = pool.starmap(create_position_maps, pool_input)
direction_data = [ent for sublist in direction_data for ent in sublist]

# Create energy binning scheme
p_max, p_min = 0, sys.maxsize

# Determine max and min energy
for item in direction_data:
    if item[2] < p_min:
        p_min = item[2]
    elif item[2] > p_max:
        p_max = item[2]

# Create bins
num_bins = args.bins
bin_sizes = np.logspace(np.log10(p_min * 0.99), np.log10(p_max * 1.001), num_bins)

# Create a sky map for each bin, for weighing by energy
initial_maps = np.zeros((num_bins, npix))
final_maps = np.zeros((num_bins, npix))
reweighed_maps = np.zeros((num_bins, npix))
reweighed_particles = [[] for i in range(npix)]

# Physical cosmic ray distribution goes with E^(-2.7), ours goes with E^(-1)
g = args.phys_index
power = args.model_index

# Populate initial and final maps
for item in direction_data:
    initial_pixel = item[0]
    final_pixel = item[1]
    p = item[2]
    p_bin = 0
    for i in range(num_bins):
        if p >= bin_sizes[i]:
            p_bin += 1
        else:
            break
    initial_maps[p_bin][initial_pixel] += 1
    final_maps[p_bin][final_pixel] += 1

# Go back through the data and reweigh the initial map. Save the individual particle data fot statistical testing
for item in direction_data:
    initial_pixel = item[0]
    final_pixel = item[1]
    p = item[2]
    p_bin = 0
    for i in range(num_bins):
        if p >= bin_sizes[i]:
            p_bin += 1
        else:
            break
    dipole_weight = 1 + 0.001 * cos_dipole_f(nside, final_pixel)
    direction_weight = final_maps[p_bin][final_pixel]
    momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1], g, power)
    reweighed_maps[p_bin][initial_pixel] += momentum_weight * dipole_weight / direction_weight
    reweighed_particles[initial_pixel].append([p, momentum_weight * dipole_weight / direction_weight])

reweighed_particles = np.array(reweighed_particles)

# Save maps and bins
prefix = args.prefix % args_dict + 'nside=' + str(nside) + 'num_bins=' + str(num_bins)
output_name = args.outdir + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name,
                    initial=initial_maps,
                    final=final_maps,
                    reweighed=reweighed_maps,
                    particles=reweighed_particles,
                    bins=bin_sizes,
                    phys_index=g,
                    model_index=power
                    )
