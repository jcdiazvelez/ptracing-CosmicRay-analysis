import numpy as np
import matplotlib.pyplot as plt

def load_chi2_from_npz(file_path):
    """
    Loads the chi2sum array from a .npz file.

    Parameters:
    - file_path (str): Path to the .npz file.

    Returns:
    - numpy.ndarray: The loaded chi2sum array.
    """
    data = np.load(file_path)
    print("Keys in file:", data.files)
    if "chi_squared" in data:
        return data["chi_squared"]
    else:
        raise KeyError("Key 'chi2sum' not found in the .npz file.")

def plot_chi2_vs_pixel(chi2sum, title='Chi² vs Pixel index', save_path=None):
    """
    Plots chi2sum values against pixel index to identify anomalies or trends.

    Parameters:
    - chi2sum (np.ndarray): Array of chi² values per pixel.
    - title (str): Title of the plot.
    - save_path (str or None): If specified, saves the figure to this path.
    """
    pixel_indices = np.arange(len(chi2sum))

    # Print summary statistics
    print(f"Total pixels: {len(chi2sum)}")
    print(f"NaN values: {np.sum(np.isnan(chi2sum))}")
    print(f"Inf values: {np.sum(np.isinf(chi2sum))}")

    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(pixel_indices, chi2sum, '.', markersize=2, alpha=0.7)
    plt.xlabel("Pixel index")
    plt.ylabel(r"$\chi^2$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    plt.show()


# Load the chi2sum array from file
file_maps_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/'
figs_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/Avance_VII/Chi2sanityCheck_realmap_pwrind-2p6_all-wei_eq_coord_norm_nside16_ang5_dof10_gaussian_diff-ord-3TeV'
chi2sum = load_chi2_from_npz(file_maps_dir+"chi2fullmap_realmap_pwrind-2p6_all-wei_eq_coord_norm_nside16_ang5_dof10_gaussian_diff-ord-3TeV.npz")
# chi2sum = load_chi2_from_npz("/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/chi2_realmap_pwrind-2p6_all-wei_nside32_pix1_dof10_gaussian-1TeV.npz")

# Plot the values
plot_chi2_vs_pixel(chi2sum, title="Chi² value per HEALPix pixel", save_path=figs_dir)
