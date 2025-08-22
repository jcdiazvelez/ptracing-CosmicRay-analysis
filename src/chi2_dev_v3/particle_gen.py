#!/usr/local/bin/python

import glob
import numpy as np
import healpy as hp
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm

class PathSegment:
    def __init__(self, datum):
        self.trackId = int(datum[0])
        self.time = datum[1]
        self.x = datum[2]
        self.y = datum[3]
        self.z = datum[4]
        self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.px = datum[5]
        self.py = datum[6]
        self.pz = datum[7]
        self.p = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        self.pid = int(datum[8])
        self.weight = datum[9]
        try:
            self.Bx = datum[10]
            self.By = datum[11]
            self.Bz = datum[12]
            self.B = np.sqrt(self.Bx ** 2 + self.By ** 2 + self.Bz ** 2)
            self.status = datum[13]
        except:
            self.Bx = 0
            self.By = 0
            self.Bz = 0
            self.B = 1
            self.status = None

    def __repr__(self):
        s = "track: %s\n" % self.trackId
        s += "\t time: %s\n" % self.time
        s += "\t position: (%s,%s,%s)\n" % (self.x, self.y, self.z)
        s += "\t momentum: (%s,%s,%s)\n" % (self.px, self.py, self.pz)
        s += "\t distance from the Sun: (%s)\n" % self.r
        s += "\t PID: %s\n" % self.pid
        s += "\t weight: %s" % self.weight
        return s

    def pitch_angle(self):
        return np.arccos((self.px * self.Bx + self.py * self.By + self.pz * self.Bz) / (self.p * self.B))


def process_particle_data_equatorial_coord(filename, nside, radius):
    file = np.load(filename)
    data_array = []

    # Rotation matrix: NICKS coordinates → ecliptic system
    NickEcl = np.array([
        [-0.20237267,  0.97163923,  0.12232136],
        [-0.97929205, -0.20005855, -0.03104294],
        [-0.00569111, -0.12607058,  0.99200495]
    ])

    # Healpy rotator: ecliptic → celestial (equatorial)
    rot = hp.Rotator(coord=['E', 'C'])

    for key in file:
        try:
            particle = file[key]
            if len(particle) < 3:
                #print(f"Key {key}: too few steps ({len(particle)})")
                raise Exception("Invalid trace (too short)")

            last_status = PathSegment(particle[-1]).status
            if last_status == -1:
                print(f"Key {key}: last segment has invalid status (-1)")
                raise Exception("Invalid trace (status)")

            # Reject if too few points or invalid status
            if len(particle) < 3 or PathSegment(particle[-1]).status == -1:
                raise Exception("Invalid trace")

            # First point of the trajectory (near origin)
            p_first = PathSegment(particle[0])
            p_last = None

            # Find the last point that crosses the radius threshold
            for i in range(len(particle) - 1, 1, -1):
                p_i = PathSegment(particle[i])
                p_j = PathSegment(particle[i - 1])
                if p_i.r > radius > p_j.r:
                    p_last = p_i
                    break

            if p_last is None:
                raise Exception("No valid exit point at radius")

            # Extract momentum vectors at initial and final states
            v_first = np.array([p_first.px, p_first.py, p_first.pz])
            v_last = np.array([p_last.px, p_last.py, p_last.pz])

            # Rotate vectors to ecliptic coordinates
            v_first_ecl = NickEcl @ v_first
            v_last_ecl = NickEcl @ v_last

            # Rotate to equatorial (celestial) coordinates
            v_first_eq = rot(v_first_ecl)
            v_last_eq = rot(v_last_ecl)

            # Normalize vectors (required for healpy pixel conversion)
            v_first_eq /= np.linalg.norm(v_first_eq)
            v_last_eq  /= np.linalg.norm(v_last_eq)

            # Convert direction vectors to HEALPix pixel numbers
            initial_pixel = hp.vec2pix(nside, *v_first_eq)
            final_pixel = hp.vec2pix(nside, *v_last_eq)

            # Extract magnetic field vector at final state
            b_local = np.array([p_last.Bx, p_last.By, p_last.Bz])

            # Rotate magnetic field vector to ecliptic coordinates
            b_ecl = NickEcl @ b_local

            # Rotate magnetic field vector to equatorial (celestial) coordinates
            b_eq = rot(b_ecl)

            # Bx_eq, By_eq, Bz_eq = b_eq

            # Save output: initial pixel, final pixel, final momentum magnitude, and magnetic field components
            data_array.append((
                initial_pixel,
                final_pixel,
                p_last.p,
                *b_eq
            ))

        except Exception as e:
            #print(f"Error processing key {key}: {e}")
            continue

    return data_array

# Parser for command line arguments
parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, default='../../data/particles/', help="Output directory for particle data")
parser.add_argument("-p", "--path", type=str, default="../../data/raw/", help="Path to data")
parser.add_argument("-N", "--nside", type=int, default=16, help="Plot resolution")
parser.add_argument("-r", "--radius", type=int, default=50000, help="Termination radius")
parser.add_argument("-g", "--phys_index", type=float, default=-1, help="Power law index for physical cosmic ray distribution")
parser.add_argument("-P", "--model_index", type=float, default=-1.0, help="Power law index for modeled cosmic ray distribution")
parser.add_argument("-t", "--threads", type=int, default=16, help="Number of simultaneous threads/processes to run in parallel")

args = parser.parse_args()
args_dict = vars(args)

nside = args.nside
radius = args.radius
path = args.path
files = sorted(glob.glob(path + "/*.npz"))  # Get all input files
n_files = len(files)

# Prepare input list for multiprocessing
pool_input = [(f, nside, radius) for f in files]

# Helper function to show progress with multiprocessing
def process_with_progress(pool, func, inputs):
    results = []
    # Use tqdm to show progress as files are processed
    for r in tqdm(pool.starmap(func, inputs), total=len(inputs), desc="Processing files"):
        results.append(r)
    return results

# Execute parallel processing
with Pool(processes=args.threads) as pool:
    direction_data = process_with_progress(pool, process_particle_data_equatorial_coord, pool_input)

# Flatten the list of results
direction_data = np.array([ent for sublist in direction_data for ent in sublist])

# Save processed data
prefix = 'eq_coord_nside=' + str(nside)
output_name = args.output + prefix
print("saving %s" % output_name)
np.savez_compressed(output_name, particles=direction_data)