#!/usr/bin/env python3
"""
Chi² Calculation Script
-----------------------
This script processes cosmic ray particle data to compute Chi² maps and
relative intensity maps from reweighted particle distributions.

Inputs:
- Reweighted particle file (.npz format)
- Energy limits (Gaussian cut around mean energy)
- Angular radius for ON/OFF regions
- Degrees of freedom for Chi² histograms
- Output directory for results

Outputs:
- Compressed .npz files containing Chi² and Rint maps
- Optional plots (Chi² skymap, Chi² PDF)
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import math
import scipy.stats


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data(file_path):
    """Load reweighted particle data from .npz file."""
    try:
        with np.load(file_path) as data:
            return data['reweighed_particles']
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Distributions
# -------------------------------------------------------------------
def get_disc_distribution(pixel_number, pixel_list, nside, ang):
    """Get ON-region distribution (disc around pixel)."""
    vec = hp.pix2vec(nside, pixel_number)
    disc_radius = ang * math.pi / 180
    pixel_disc = hp.query_disc(nside, vec, disc_radius, inclusive=True)
    return _get_sky_distribution(pixel_list[pixel_disc]), len(pixel_disc)


def get_off_pixels_distribution(pixel_number, pixel_list, nside, ang):
    """Get OFF-region distribution (all sky excluding ON disc)."""
    vec = hp.pix2vec(nside, pixel_number)
    disc_radius = ang * math.pi / 180
    on_pixels = hp.query_disc(nside, vec, disc_radius, inclusive=True)
    all_pixels = np.arange(len(pixel_list))
    off_pixels = np.setdiff1d(all_pixels, on_pixels)
    return _get_sky_distribution(pixel_list[off_pixels]), len(off_pixels)


def _get_sky_distribution(pixel_list):
    """Aggregate energies and weights from a list of pixels."""
    energies, weights = [], []
    for pixel in pixel_list:
        for e, w in pixel:
            energies.append(e)
            weights.append(w)
    return np.array([energies, weights])


def impose_energy_range(distribution, min_energy, max_energy):
    """Filter distribution to the specified energy range."""
    energies, weights = distribution
    mask = (energies >= min_energy) & (energies <= max_energy)
    return np.array([energies[mask], weights[mask]])


# -------------------------------------------------------------------
# Chi² calculations
# -------------------------------------------------------------------
def observational_weight(particle_energy, obs_parameters):
    """Gaussian observational weight in log-energy space."""
    if obs_parameters[0] == -1 and obs_parameters[1] == -1:
        return 1.0
    sigma, mid = obs_parameters
    log_e = np.log10(particle_energy)
    return np.exp(-0.5 * ((log_e - np.log10(mid)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def chi2_and_rint_map(data1, data2, wei1, wei2, npix1, npix2, dof=10):
    """Compute reduced Chi² and relative intensity."""
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    ebins = np.logspace(np.log10(min_val), np.log10(max_val), dof + 1)

    # Apply observational weights
    wei1 = wei1 * observational_weight(data1, [0.25, 7e3])
    wei2 = wei2 * observational_weight(data2, [0.25, 7e3])
    norm1, norm2 = np.sum(wei1), np.sum(wei2)
    wei1 /= norm1
    wei2 /= norm2

    # Histograms
    N_on, _ = np.histogram(data1, bins=ebins)
    N_off, _ = np.histogram(data2, bins=ebins)
    Wi_on, _ = np.histogram(data1, bins=ebins, weights=wei1)
    Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2)
    S2i_on, _ = np.histogram(data1, bins=ebins, weights=wei1**2)
    S2i_off, _ = np.histogram(data2, bins=ebins, weights=wei2**2)

    # Variance per bin
    valid = (Wi_on > 0) & (Wi_off > 0)
    di2 = np.full_like(Wi_on, np.inf, dtype=np.float64)
    di2[valid] = Wi_off[valid] * (S2i_on[valid] / Wi_on[valid] + S2i_off[valid] / Wi_off[valid])

    mask = (N_on > 20) & (N_off > 20) & (di2 > 0) & np.isfinite(di2)
    if np.sum(mask) == 0:
        return np.nan, np.nan

    chi2_red = np.sum((Wi_on[mask] - Wi_off[mask])**2 / di2[mask]) / (len(ebins) - 1)
    Rint = (norm1 / norm2) * (npix2 / npix1) - 1
    return chi2_red, Rint


def perform_Chi2_and_Rint(particles, limits, ang, dof=10, progress=False):
    """Compute Chi² and relative intensity for all pixels."""
    npix = len(particles)
    nside = hp.npix2nside(npix)
    chi2sum, Rint = [], []

    loop = tqdm(range(npix), desc="Computing χ²", unit="pix") if progress else range(npix)
    for i in loop:
        off_dist, npix2 = get_off_pixels_distribution(i, particles, nside, ang)
        on_dist, npix1 = get_disc_distribution(i, particles, nside, ang)
        off_dist = impose_energy_range(off_dist, limits[0], limits[1])
        on_dist = impose_energy_range(on_dist, limits[0], limits[1])
        chi2, rint = chi2_and_rint_map(on_dist[0], off_dist[0], on_dist[1], off_dist[1], npix1, npix2, dof)
        chi2sum.append(chi2)
        Rint.append(rint)
    return np.array(chi2sum), np.array(Rint)


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
def plot_skymap(skymap, title, label='', filename=None):
    """Plot a Mollweide skymap."""
    hp.mollview(skymap, title=title, unit=label, cmap="coolwarm")
    if filename:
        plt.savefig(filename, dpi=250)
        plt.close()


def chi2_pdf_plot(chi2_vals, dof=10, save_path=None):
    """Plot empirical χ² PDF vs theoretical χ² distribution."""
    xbins = np.logspace(-2, 3, 100)
    hist, edges = np.histogram(chi2_vals[~np.isnan(chi2_vals)], bins=xbins, density=True)
    rv = scipy.stats.chi2(dof)

    plt.figure(figsize=(8, 6))
    plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, label=f"Theory χ² PDF (dof={dof})")
    plt.plot(edges[1:], hist, label="Empirical χ²")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("χ²"); plt.ylabel("PDF")
    plt.legend(); plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Compute Chi² maps from reweighted particles")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input .npz file with reweighted particles")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for Chi² maps")
    parser.add_argument("-a", "--ang", type=float, default=5,
                        help="Angular radius (degrees)")
    parser.add_argument("-d", "--dof", type=int, default=10,
                        help="Degrees of freedom for χ² histogram")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress bar")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load data
    particles = load_data(args.input)
    if particles is None:
        exit(1)

    # Energy limits (Gaussian cut in log10(E))
    sigma = 0.25
    mean_gauss = np.log10(7e3)
    limits = [10**(mean_gauss - 3*sigma), 10**(mean_gauss + 3*sigma)]

    # Compute Chi² and Rint maps
    chi2, Rint = perform_Chi2_and_Rint(particles, limits, args.ang, args.dof, progress=args.progress)

    # Save results
    np.savez_compressed(f"{args.output}/Chi2_map.npz", chi_squared=chi2)
    np.savez_compressed(f"{args.output}/Rint_map.npz", relative_intensity=Rint)
    logger.info(f"Saved Chi² and Rint maps to {args.output}")

# python chi2_calculation.py \
#     -i ../../data/particles/weights_nside=16.npz \
#     -o ../../data/maps/ \
#     -a 5 \
#     -d 10 \
#     --progress --verbose
