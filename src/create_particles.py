#!/usr/local/bin/python

import glob
import numpy as np
import healpy as hp
from data_methods import process_particle_data
from multiprocessing import Pool
from argparse import ArgumentParser

# Parser for reading command line arguments
parser = ArgumentParser()

parser.add_argument("-o", "--output", type=str, default='../data/',
                    help="Output directory for particle data")
parser.add_argument("-p", "--path", type=str, default="../../data/newsets/h5hybrid-2",
                    help="Path to data")
parser.add_argument("-N", "--nside", type=int, default="16",
                    help="plot resolution")
parser.add_argument("-r", "--radius", type=int, default="50000",
                    help="termination radius")
parser.add_argument("-g", "--phys_index", type=float, default="-2.7",
                    help="power law index for physical cosmic ray distribution")
parser.add_argument("-P", "--model_index", type=float, default="-1.0",
                    help="power law index for modelled cosmic ray distribution")
parser.add_argument("-t", "--threads", type=int, default=16,
                    help="Number of simulataneous threas/processes to run in parallel")

args = parser.parse_args()
args_dict = vars(args)

# Set map parameters
nside = args.nside
npix = hp.nside2npix(nside)
radius = args.radius
path = args.path

# Prepare filenames
filename = "*.npz"
path = path + "/" + filename

# Prepare file names for processing
files = sorted(glob.glob(path))
n_files = len(files)

# Use 16 worker processes
pool = Pool(processes=args.threads)

# Create pool input for direction data map
pool_input = []
for i in range(n_files):
    pool_input.append((files[i], nside, radius))

# Generate and flatten direction data
direction_data = pool.starmap(process_particle_data, pool_input)
direction_data = np.array([ent for sublist in direction_data for ent in sublist])

# Save produced data
prefix = 'nside=' + str(nside)
output_name = args.output + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name, particles=direction_data)
