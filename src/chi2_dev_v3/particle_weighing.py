#!/usr/bin/env python3
"""
Particle Weighing Script
------------------------
This script processes cosmic ray particle data to generate weight maps
based on:
    - energy distributions
    - directional factors
    - observational Gaussian weights
    - imposed uniform/dipole weights

Inputs:
- HEALPix resolution parameter (nside)
- Number of energy bins
- Imposed weight parameters (uniform, dipole factor)
- Physical power-law index
- Particle file (.npz) containing pixel assignments and energies
- Output file path

Outputs:
- Compressed .npz file with reweighted particle data
- Weight arrays including momentum, direction, observational, and imposed factors
"""

import numpy as np
import healpy as hp
import logging
from argparse import ArgumentParser
from tqdm import tqdm


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Weighting helper functions
# -------------------------------------------------------------------
def cos_dipole_f(nside, pix, bx, by, bz):
    """
    Compute cosine between dipole vector (bx, by, bz) and HEALPix pixel direction.
    """
    pxf, pyf, pzf = hp.pix2vec(nside, pix)
    return -(pxf * bx + pyf * by + pzf * bz) / \
           (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1e-16) / \
           np.sqrt(bx * bx + by * by + bz * bz)


def powerlaw_pdf(x, x_min, x_max, power):
    """
    Probability density function of a power-law distribution.
    """
    if power == -1.0:
        return x**power / np.log(x_max / x_min)
    else:
        x_min_g, x_max_g = x_min ** (power + 1.), x_max ** (power + 1.)
        return (power + 1.) / (x_max_g - x_min_g) * x**power


def weight_powerlaw(x, x_min, x_max, exponent, power):
    """
    Compute weight of a value based on a power-law PDF.
    """
    return x**exponent / powerlaw_pdf(x, x_min, x_max, power)


def observational_weight(particle_energy, obs_parameters):
    """
    Compute observational weights based on Gaussian distribution in log-energy space.
    """
    if obs_parameters[0] == -1 and obs_parameters[1] == -1:
        return 1.0
    sigma = obs_parameters[0]
    mid_energy = np.log10(obs_parameters[1])
    logged_energy = np.log10(particle_energy)
    return np.exp(-0.5 * ((logged_energy - mid_energy) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# -------------------------------------------------------------------
# Core processing function
# -------------------------------------------------------------------
def compute_particle_weights(nside, bins, imposed_parameters, physical_index, energy,
                             particle_file, progress=False):
    """
    Process particle file and compute weights.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    bins : int
        Number of logarithmic energy bins.
    imposed_parameters : list [uniform, dipole_factor]
        Parameters for imposed weights.
    physical_index : float
        Power-law index for physical distribution.
    energy_range : float
        Energy range.
    particle_file : str
        Input .npz file with particle data.
    progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    np.ndarray
        Array of reweighted particles grouped by initial pixel.
    """
    try:
        particles_data = np.load(particle_file, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"File not found: {particle_file}")
        return None
    except Exception as e:
        logger.error(f"Error loading {particle_file}: {e}")
        return None

    particles = particles_data['particles']
    npix = hp.nside2npix(nside)

    # Extract particle energies and build logarithmic bins
    energies = particles[:, 2]
    #energy_bins = np.logspace(np.log10(min(energies)), np.log10(max(energies)), bins + 1)
    energy_bins = np.logspace(np.log10(energy[0]), np.log10(energy[1]), bins + 1)
    print('Imposed energy range', energy[0], energy[1])
    print('Max energy range', min(energies), max(energies))
    logger.info(f"Max energy range: {min(energies):.2e} – {max(energies):.2e} GeV")

    # Initialize storage
    final_maps = np.zeros((bins, npix))
    energy_bin_counts = np.zeros((npix, bins))
    pixel_counts = np.zeros(npix)

    # Count particles per pixel and bin
    loop_iter = tqdm(particles, desc="Counting bins", unit="particle") if progress else particles
    for item in loop_iter:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = np.digitize(p, energy_bins) - 1
        pixel_counts[final_pixel] += 1.0
        if 0 <= p_bin < bins:
            energy_bin_counts[final_pixel, p_bin] += 1.0

    # Normalize per-pixel weights
    for ipix in range(npix):
        eweight = 1.0 / energy_bin_counts[ipix]
        eweight[np.isinf(eweight)] = 0.0
        eweight_norm = np.sum(eweight)
        if eweight_norm > 0:
            eweight /= eweight_norm
        final_maps[:, ipix] = eweight

    # Assign weights to each particle
    reweighed_particles = [[] for _ in range(npix)]
    loop_iter = tqdm(particles, desc="Assigning weights", unit="particle") if progress else particles
    for item in loop_iter:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        p = item[2]
        bx, by, bz = item[3], item[4], item[5]

        p_bin = np.digitize(p, energy_bins) - 1
        direction_weight = final_maps[p_bin, final_pixel] if 0 <= p_bin < bins else 0

        imposed_weight = 1.0 + imposed_parameters[1] * cos_dipole_f(nside, final_pixel, bx, by, bz)
        momentum_weight = weight_powerlaw(p, energy_bins[0], energy_bins[-1], physical_index, -1)

        total_weight = momentum_weight * imposed_weight * direction_weight
        reweighed_particles[initial_pixel].append([p, total_weight])

    return convert_to_numpy_array(reweighed_particles)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def convert_to_numpy_array(reweighed_particles):
    """
    Convert jagged list of weighted particles to a padded NumPy array.
    """
    max_length = max(len(sublist) for sublist in reweighed_particles)
    padded_array = np.array(
        [sublist + [[0, 0]] * (max_length - len(sublist)) for sublist in reweighed_particles],
        dtype=np.float32
    )
    return padded_array


def save_results(reweighed_particles_array, output_file):
    """
    Save results as compressed NumPy .npz file.
    """
    np.savez_compressed(output_file, reweighed_particles=reweighed_particles_array)
    logger.info(f"Results saved to {output_file}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Compute cosmic ray particle weights")
    parser.add_argument("-p", "--particles", type=str, required=True,
                        help="Path to input particle .npz file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to save reweighted output .npz")
    parser.add_argument("-N", "--nside", type=int, default=16,
                        help="HEALPix resolution (power of 2)")
    parser.add_argument("-b", "--bins", type=int, default=120,
                        help="Number of logarithmic energy bins")
    parser.add_argument("-i", "--index", type=float, default=-2.6,
                        help="Physical power-law index")
    parser.add_argument("-e", "--energy", nargs=2, type=float, default=[6e3, 18e3],
                        help="Energy range")
    parser.add_argument("--imposed", nargs=2, type=float, default=[1.0, 0.001],
                        metavar=("UNIFORM", "DIPOLE"),
                        help="Imposed weight parameters: uniform, dipole factor")
    parser.add_argument("--progress", action="store_true",
                        help="Enable progress bars")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Compute weights
    result = compute_particle_weights(args.nside, args.bins, args.imposed,
                                      args.index, args.energy, args.particles, progress=args.progress)

    if result is not None:
        save_results(result, args.output)
        logger.info("Processing completed successfully.")
    else:
        logger.error("Processing failed.")

# python particle_weighing.py \
#     -p ../../data/particles/eq_coord_nside=16.npz \
#     -o ../../data/particles/weights_nside=16_newPlan.npz \
#     -N 16 \
#     -b 120 \
#     -i -2.6 \
#     -e 6e3 18e3 \
#     --imposed 1.0 0.01 \
#     --progress --verbose

