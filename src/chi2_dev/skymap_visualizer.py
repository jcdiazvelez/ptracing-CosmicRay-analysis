
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import pylab

def load_chi2_from_npz(file_path, key="chi_squared"):
    data = np.load(file_path)
    print("Keys in file:", data.files)
    if key in data:
        return data[key]
    else:
        raise KeyError(f"Key '{key}' not found in {file_path}")

def plot_skymap(skymap, title, proj='C', label='', filename=None, 
                thresh=None, dMin=None, dMax=None, sun=None):
    """
    Plots a skymap using the HEALPix Mollweide projection, allowing customization 
    of color scales, annotations, and projections.
    
    Parameters:
    - skymap (ndarray): The data to visualize, typically a 1D array representing pixel values.
    - title (str): Title for the skymap plot.
    - proj (str or tuple): Projection type or rotation tuple.
      - 'C': Central projection (default).
      - 'C0': Centered at 0° longitude.
      - Tuple (lon, lat, psi) for custom rotations.
    - label (str): Label for the color bar (unit of the data).
    - filename (str, optional): Path to save the plotted figure. If None, the figure is not saved.
    - thresh (float, optional): Not used directly but could be for masking or thresholding the data.
    - dMin, dMax (float, optional): Minimum and maximum data values for color scaling.
    - sun (tuple, optional): Coordinates (longitude, latitude) to mark the Sun's position on the map.
    
    This function is widely used in cosmic ray analysis and skymap visualizations, 
    leveraging HEALPix's hp.mollview for spherical data projection.
    """
    
    # Configure global plot parameters for font size and style
    params = {'legend.fontsize': 'x-large',
              'axes.titlesize': '20',
              "font.family": "serif"}
    pylab.rcParams.update(params)

    # Define the colormap for the skymap visualization
    colormap = plt.get_cmap("coolwarm")
    
    # Set the color for under-range values to white if applicable
    if hasattr(colormap, 'set_under'):
        colormap.set_under("w")

    # Determine the rotation based on the projection type
    rotation = (0, 0, 0) if proj == 'C0' else (-180, 0, 0) if proj == 'C' else proj

    # Plot the sky map using the Mollweide projection
    hp.mollview(skymap, title=title, rot=rotation, unit=label,
                margins=(0.0, 0.03, 0.0, 0.13), notext=False,
                cmap=colormap, min=dMin, max=dMax)
    
    # Get the current figure and its main axis
    fig = plt.gcf()
    ax = fig.get_axes()[0] if fig.get_axes() else None
    
    # Add annotations for 0° and 360° at the edges of the map
    if ax and proj in ['C', 'C0']:
        ax.annotate(r"0$^\circ$", xy=(1.8, 0.625), size="x-large")
        ax.annotate(r"360$^\circ$", xy=(-1.95, 0.625), size="x-large")
    
    # Highlight the Sun's position on the map if coordinates are provided
    if sun is not None:
        hp.projscatter(sun[0], sun[1], lonlat=True, coord='C')
        hp.projtext(sun[0], sun[1], 'Sun', lonlat=True, coord='C', fontsize=18)
    
    # Overlay a grid to provide latitude and longitude references
    hp.graticule()

    # Save the figure to a file if a filename is provided
    if filename:
        fig.savefig(filename, dpi=250)
        plt.close(fig)  # Free memory by closing the figure

def plot_chi_squared(chi_squared_map, out_dir, name):
    """
    Plots a skymap of Chi² values using the Mollweide projection.
    
    Parameters:
    - chi_squared_map (ndarray): Array of Chi² values for each pixel in the skymap.
    - out_dir (str): Directory path where the plot image will be saved.
    - name (str): Name of the output file (without extension).
    
    This function visualizes the Chi² distribution over the sky, highlighting 
    regions with the highest and lowest Chi² values.
    """
    
    chi2sum = chi_squared_map
    
    # Print the maximum and minimum Chi² values along with their labels
    print('chi2sum[np.argmax(chi2sum)] ' + name, chi2sum[np.argmax(chi2sum)])
    print('chi2sum[np.argmin(chi2sum)] ' + name, chi2sum[np.argmin(chi2sum)])
    
    # Prepare data for plotting by taking the absolute values of Chi²
    z_values = np.abs(chi2sum)
    
    # Generate the skymap visualization
    plot_skymap(z_values,
                title=name,
                label="Range",
                proj='C0',
                dMin=0.0,
                dMax=400.0,
                # dMin=chi2sum[np.argmin(chi2sum)],
                # dMax=chi2sum[np.argmax(chi2sum)],
                filename=out_dir + name)
    
    # Close the plot to free memory
    plt.close()

# Define the output directory for plots and results
file_plot_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/'
file_maps_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/'
chi2sum = load_chi2_from_npz(file_maps_dir+"chi2_realmap_pwrind-2p6_all-wei_nside32_pix1_dof10_wo-gaussian.npz", key="chi_squared")
plot_chi_squared(chi2sum, file_plot_dir, 'skymap_chi2_real-mapping_pwrind=-2p6_all-weights_nside=32_pix1_dof10_wo-gaussian_chi2range-0-400')