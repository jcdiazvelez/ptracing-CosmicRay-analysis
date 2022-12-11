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

from data_methods import run_file, reweigh_file

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

pool = Pool(processes=4)  # start 4 worker processes

pool_input = []
for i in range(nfiles):
    pool_input.append((files[i], nside))

data_arrays = pool.starmap(run_file, pool_input)

data_arrays = [ent for sublist in data_arrays for ent in sublist]

npix = hp.nside2npix(nside)

initial_map = np.zeros(npix, dtype=np.double)
final_map = np.zeros(npix, dtype=np.double)

for i in range(len(data_arrays)):
    if data_arrays[i][0] < 0:
        final_map[-data_arrays[i][0] - 1] += data_arrays[i][1]
    else:
        initial_map[data_arrays[i][0] - 1] += data_arrays[i][1]

hp.visufunc.mollview(initial_map)
plt.title('Initial Momenta')
plt.savefig('./figs/initial')

hp.visufunc.mollview(final_map)
plt.title('Final Positions')
plt.savefig('./figs/final')

pool_input = []
for i in range(nfiles):
    pool_input.append((files[i], nside, final_map))

reweighed_array = pool.starmap(reweigh_file, pool_input)
reweighed_array = [ent for sublist in reweighed_array for ent in sublist]

reweighed_map = np.zeros(npix, dtype=np.double)

for i in range(len(reweighed_array)):
    reweighed_map[reweighed_array[i][0]] += 1.0 / reweighed_array[i][1]

hp.visufunc.mollview(reweighed_map)
plt.title('Reweighed Initial Momenta')
plt.savefig('./figs/reweighed')
