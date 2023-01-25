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
parser.add_argument("-j", "--jobnumber", type=int, default=0,
                    help="Index of job (for selecting subset of files)")
parser.add_argument("-n", "--nfiles", type=int, default=0,
                    help="Number of files per job (for selecting subset of files)")
parser.add_argument("-f", "--fieldtype", type=str, default="uniform",
                    help="Magnetic field type")
parser.add_argument("-p", "--path", type=str,
                    # default="/data/sim/solarmodel/ptracing/h5mdh/sets/",
                    default="/data/sim/solarmodel/ptracing/jc/sets/backtrack3/",
                    help="Path to data")
parser.add_argument("-d", "--direction", type=str, default="back",
                    help="direction of propagation")
parser.add_argument("-s", "--subtract-first", default=False, action='store_true',
                    help="direction of propagation")
parser.add_argument("-o", "--outdir", type=str, default=".",
                    help="output directory")
parser.add_argument("-N", "--nside", type=int, default="30",
                    help="plot resolution")
parser.add_argument("--prefix", type=str, default="a_crossings_%(fieldtype)s_energy_",
                    help="output directory")
parser.add_argument("-r", "--radius", type=int, default="50000",
                    help="termination radius")

args = parser.parse_args()
args_dict = vars(args)

# File names for particle data
filename = "%s*.npz" % args.fieldtype
path = args.path + "/" + filename

print(path)

files = sorted(glob.glob(path))
n_files = len(files)

print(n_files, "total files")

if args.nfiles:
    n_files = args.nfiles

print("processing", n_files, " files")

# Set healpy nside
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
num_bins = 10
bin_sizes = np.logspace(np.log10(p_min * 0.99), np.log10(p_max * 1.001), num_bins)

# Create a sky map for each bin, for weighing by energy
initial_maps = np.zeros((num_bins, npix))
final_maps = np.zeros((num_bins, npix))
reweighed_maps = np.zeros((num_bins, npix))

# Physical cosmic ray distribution goes with E^(-2.7), ours goes with E^(-1)
g = -2.7
power = -1

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

# Go back through the data and reweigh the initial map
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

# Save maps and bins
prefix = args.prefix % args_dict
output_name = '../data/' + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name,
                    initial=initial_maps,
                    final=final_maps,
                    reweighed=reweighed_maps,
                    bins=bin_sizes
                    )
