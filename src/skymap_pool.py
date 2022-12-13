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

from data_methods import create_position_maps, perform_weighting

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

args = parser.parse_args()

args_dict = vars(args)

start_time = time.time()

directory = args.fieldtype
if args.fieldtype == "helio":
    directory = "heliosphere"
direction = args.direction + "track"

filename = "%s*.npz" % args.fieldtype
path = args.path + "/" + filename

print(path)

files = sorted(glob.glob(path))
nfiles = len(files)

print(nfiles, "total files")

if args.nfiles:
    nfiles = args.nfiles

print("processing", nfiles, " files")

nside = args.nside
npix = hp.nside2npix(nside)

pool = Pool(processes=16)  # start 4 worker processes

pool_input = []
for i in range(nfiles):
    pool_input.append((files[i], nside))

direction_data = pool.starmap(create_position_maps, pool_input)
direction_data = [ent for sublist in direction_data for ent in sublist]

p_max, p_min = 0, sys.maxsize

for item in direction_data:
    if item[2] < p_min:
        p_min = item[2]
    elif item[2] > p_max:
        p_max = item[2]

print(p_min)
print(p_max)

num_bins = 10
bin_sizes = np.logspace(np.log10(p_min), np.log10(p_max), num_bins)

print(bin_sizes)

initial_maps = np.zeros((num_bins, npix))
final_maps = np.zeros((num_bins, npix))
reweighed_maps = np.zeros((num_bins, npix))

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

print(initial_maps)
print(final_maps)

pool_input = []

for i in range(nfiles):
    pool_input.append((files[i], nside, final_maps, bin_sizes))

reweighed_data = pool.starmap(perform_weighting, pool_input)
reweighed_data = [ent for sublist in reweighed_data for ent in sublist]

for item in reweighed_data:
    pixel = item[0]
    positional_weight = item[1]
    p_bin = item[2]

    reweighed_maps[p_bin][pixel] += positional_weight

prefix = args.prefix % args_dict

output_name = '../data/' + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name,
                    initial=initial_maps,
                    final=final_maps,
                    reweighed=reweighed_maps
                    )
