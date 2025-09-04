#!/usr/bin/env python3
"""
chi2_plot.py

Script to visualize Chi2 and Rint maps produced by chi2_calculation.py.
Generates Healpy sky maps and a Chi2 vs Rint scatter plot.

Usage:
    python chi2_plot.py \
        -c ../../data/chi2/Chi2_map.npy \
        -r ../../data/chi2/Rint_map.npy \
        -o ../../figures/chi2_plots/
"""

import os
import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# =========================
# Helper function to save mollview plots
# =========================
def plot_healpy_map(data, title, outfile, unit=""):
    """
    Plot a Healpy Mollweide map and save as PNG.
    """
    hp.mollview(
        data,
        title=title,
        unit=unit,
        cmap="viridis",
        norm="hist",
        cbar=True
    )
    hp.graticule()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Plot Chi2 and Rint maps")
    parser.add_argument("-c", "--chi2", required=True,
                        help="Path to Chi2_map.npy")
    parser.add_argument("-r", "--rint", required=True,
                        help="Path to Rint_map.npy")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load maps
    chi2_map = np.load(args.chi2)
    rint_map = np.load(args.rint)

    # =========================
    # Plot Chi2 map
    # =========================
    chi2_out = os.path.join(args.output, "Chi2_map.png")
    plot_healpy_map(chi2_map, "Chi2 Map", chi2_out, unit="Chi2")

    # =========================
    # Plot Rint map
    # =========================
    rint_out = os.path.join(args.output, "Rint_map.png")
    plot_healpy_map(rint_map, "Rint Map", rint_out, unit="Rint")

    # =========================
    # Scatter plot Chi2 vs Rint
    # =========================
    scatter_out = os.path.join(args.output, "Chi2_vs_Rint.png")
    plt.figure(figsize=(6, 5))
    plt.scatter(rint_map, chi2_map, alpha=0.6, s=10, c="blue")
    plt.xlabel("Rint")
    plt.ylabel("Chi2")
    plt.title("Chi2 vs Rint")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_out, dpi=300)
    plt.close()

    print(f"Plots saved to {args.output}")

if __name__ == "__main__":
    main()
