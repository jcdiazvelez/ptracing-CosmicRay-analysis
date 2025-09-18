#!/usr/bin/env python3
"""
chi2_rint_plot.py

Script to visualize Rint (and optionally Chi2) maps produced by chi2_calculation.py.
Supports both .npy and .npz inputs.

Includes option for projection/rotation.

Usage examples:
    # Only Rint map, default projection
    python chi2_rint_plot.py \
        -r ../../data/chi2/Rint_map.npz \
        -o ../../figures/chi2_plots/

    # Rint + Chi2 with custom rotation
    python chi2_rint_plot.py \
        -c ../../data/chi2/Chi2_map.npz \
        -r ../../data/chi2/Rint_map.npz \
        -o ../../figures/chi2_plots/ \
        --proj "45,0,0"
"""

import os
import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


# =========================
# Helper functions
# =========================
def load_map(path, key=None):
    """
    Load a map from .npy or .npz file.
    If .npz, use the given key or fall back to the first array.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        if key and key in data:
            return data[key]
        else:
            first_key = list(data.keys())[0]
            print(f"[INFO] Using key '{first_key}' from {path}")
            return data[first_key]
    else:
        return np.load(path)


def parse_rotation(proj):
    """
    Convert projection option into a Healpy rotation tuple.
    - 'C0' → (0, 0, 0)
    - 'C'  → (-180, 0, 0)
    - Custom string 'lon,lat,psi' → tuple(float)
    """
    if proj == "C0":
        return (0, 0, 0)
    elif proj == "C":
        return (-180, 0, 0)
    else:
        try:
            return tuple(map(float, proj.split(",")))
        except Exception:
            raise ValueError(f"Invalid proj argument: {proj}")


def plot_healpy_map(data, title, outfile, unit="", rotation=(0, 0, 0)):
    """
    Plot a Healpy Mollweide map with optional rotation and save as PNG.
    """
    fig = plt.figure()
    hp.mollview(
        data,
        title=title,
        unit=unit,
        cmap="coolwarm",
        norm="hist",
        cbar=True,
        rot=rotation,
        margins=(0.0, 0.03, 0.0, 0.13),  # space for colorbar
        notext=False,
        fig=fig.number
    )
    hp.graticule()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Plot Chi2 and/or Rint maps")
    parser.add_argument("-c", "--chi2", required=False,
                        help="Path to Chi2_map file (.npy or .npz)")
    parser.add_argument("-r", "--rint", required=True,
                        help="Path to Rint_map file (.npy or .npz)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for plots")
    parser.add_argument("--proj", type=str, default="C0",
                        help="Projection/rotation (options: 'C0', 'C', or 'lon,lat,psi')")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    saved_files = []

    # Determine rotation
    rotation = parse_rotation(args.proj)
    print(f"[INFO] Using rotation {rotation}")

    # =========================
    # Plot Rint map
    # =========================
    rint_map = load_map(args.rint, key="relative_intensity")
    print(f"[DEBUG] Loaded Rint_map from {args.rint}")
    print("  shape:", rint_map.shape)
    print("  min:", np.nanmin(rint_map))
    print("  max:", np.nanmax(rint_map))
    print("  mean:", np.nanmean(rint_map))

    rint_out = os.path.join(args.output, "Rint_map_fix_v2
    .png")
    plot_healpy_map(rint_map, "Rint Map", rint_out, unit="Rint", rotation=rotation)
    saved_files.append(rint_out)

    # =========================
    # Plot Chi2 map (if provided)
    # =========================
    if args.chi2:
        chi2_map = load_map(args.chi2, key="chi2_map")
        print(f"[DEBUG] Loaded Chi2_map from {args.chi2}")
        print("  shape:", chi2_map.shape)
        print("  min:", np.nanmin(chi2_map))
        print("  max:", np.nanmax(chi2_map))
        print("  mean:", np.nanmean(chi2_map))

        chi2_out = os.path.join(args.output, "Chi2_map.png")
        plot_healpy_map(chi2_map, "Chi2 Map", chi2_out, unit="Chi2", rotation=rotation)
        saved_files.append(chi2_out)

    print(f"[INFO] Saved plots: {saved_files}")


if __name__ == "__main__":
    main()

# python chi2_rint_plot.py 
# -r '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/newPlan_Rint_map.npz' 
# -o '../../figs/results_sept_2025/' 
# --proj C