
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import pylab
import scipy.stats

def load_chi2_from_npz(file_path, key="chi_squared"):
    data = np.load(file_path)
    print("Keys in file:", data.files)
    if key in data:
        return data[key]
    else:
        raise KeyError(f"Key '{key}' not found in {file_path}")

def load_from_npz(file_path, key="particles"):
    data = np.load(file_path)
    print("Keys in file:", data.files)
    if key in data:
        return data[key]
    else:
        raise KeyError(f"Key '{key}' not found in {file_path}")
    
# Chi² PDF plot function
def chi2_pdf_plot(chi2_concat, save_path=None, dof=10):
    """
    Plots a Probability Density Function (PDF) of Chi² values and compares it 
    to the theoretical Chi² distribution.

    Parameters:
    - chi2_concat (list or array-like): List of calculated Chi² values.
    - dof (int): Degrees of freedom for the theoretical Chi² distribution.
    - save_path (str, optional): Path to save the plotted figure. If None, the plot is displayed.

    The function generates a log-log plot comparing the empirical distribution of
    Chi² values to the expected theoretical Chi² PDF.
    """
    # Define logarithmically spaced bins for the histogram
    xbins = np.logspace(-2, 3, 100)

    # Calculate the histogram of the Chi² values, normalized to form a probability density
    hist, edges = np.histogram(chi2_concat, bins=xbins, density=True)

    # Create a theoretical Chi² distribution
    rv = scipy.stats.chi2(dof)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot the theoretical Chi² PDF
    plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, 
             label=f'PDF ({dof} dof)')

    # Plot the empirical histogram of the Chi² values
    plt.plot(edges[1:], hist, label='χ² Calculado')

    # Set logarithmic scales
    plt.yscale('log')
    plt.xscale('log')

    # Add plot titles and labels
    # plt.title('Chi² Probability Density Function (PDF)')
    plt.xlabel('χ²')
    plt.ylabel('Función de densidad de probabilidad (PDF)')

    # Display legend and grid
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"PDF plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

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

def plot_chi_squared(chi_squared_map, out_dir, name, proj='C', label="χ²/v"):
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
    # chi2sum = hp.smoothing(chi2sum, fwhm=np.radians(10.0))
    
    # Print the maximum and minimum Chi² values along with their labels
    print('chi2sum[np.argmax(chi2sum)] ' + name, chi2sum[np.argmax(chi2sum)])
    print('chi2sum[np.argmin(chi2sum)] ' + name, chi2sum[np.argmin(chi2sum)])
    
    # Prepare data for plotting by taking the absolute values of Chi²
    # z_values = np.abs(chi2sum)
    
    # Generate the skymap visualization
    plot_skymap(chi2sum,
                title=None,
                label=label,
                proj=proj,
                dMin=chi2sum[np.argmin(chi2sum)],
                dMax=chi2sum[np.argmax(chi2sum)],
                filename=out_dir + name)
    
    # Close the plot to free memory
    plt.close()

def plot_distribution(wion, out_dir, name, proj='C', label=''):
    """
    Plots a skymap of distributions using the Mollweide projection.
    
    Parameters:
    - particles (ndarray): Array of distribution for each pixel in the skymap.
    - out_dir (str): Directory path where the plot image will be saved.
    - name (str): Name of the output file (without extension).
    
    This function visualizes the distribution over the sky, highlighting 
    regions with the highest and lowest values.
    """
    
    # Print the maximum and minimum Chi² values along with their labels
    print('max limit ' + name, wion[np.argmax(wion)])
    print('min limit' + name, wion[np.argmin(wion)])
    
    # Prepare data for plotting by taking the absolute values of Chi²
    # z_values = np.abs(chi2sum)

    # wion = hp.smoothing(wion, fwhm=np.radians(10.0))
    # Generate the skymap visualization
    plot_skymap(wion,
                title=None,
                label=label,
                proj=proj,
                dMin=wion[np.argmin(wion)],
                dMax=wion[np.argmax(wion)],
                filename=out_dir + name)
    
    # Close the plot to free memory
    plt.close()

# Define the input/output directory for plots and results
file_plot_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/results_july_2025/'
file_maps_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/'

# chi2 and relative intensity skymaps
chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_newOFFdist.npz", key="chi_squared")
Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_newOFFdist.npz", key="chi_squared")
# chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei.npz", key="chi_squared")
# Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei.npz", key="chi_squared")
# chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_dipAmp0p0.npz", key="chi_squared")
# Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_dipAmp0p0.npz", key="chi_squared")
# chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-7TeV.npz", key="chi_squared")
# Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-7TeV.npz", key="chi_squared")
# chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-10TeV.npz", key="chi_squared")
# Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-10TeV.npz", key="chi_squared")
# chi2 = load_chi2_from_npz(file_maps_dir+"Chi2_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-12TeV.npz", key="chi_squared")
# Rint = load_chi2_from_npz(file_maps_dir+"Rint_realmap_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-12TeV.npz", key="chi_squared")
plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_newOFFdist', 'C', "χ²/v")
plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_newOFFdist', 'C', "")
# plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_smooth_10', 'C', "χ² reducido")
# plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_smooth_10', 'C', "")
# chi2_pdf_plot(chi2,file_plot_dir+'Chi2andRint_pdf_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_smooth_10.png',10)
# plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_dipAmp0p0', 'C', "χ² reducido")
# plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_dipAmp0p0', 'C', "")
# chi2_pdf_plot(chi2,file_plot_dir+'Chi2andRint_pdf_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_woMomwei-and-obswei_woAbs_dipAmp0p0.png',10)
# plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-7TeV_woAbs_smooth_10', 'C', "χ² reducido")
# plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-7TeV_woAbs_smooth_10', 'C', "")
# chi2_pdf_plot(chi2,file_plot_dir+'Chi2andRint_pdf_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-7TeV_woAbs_smooth_10.png',10)
# plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-10TeV_woAbs_smooth_10', 'C', "χ² reducido")
# plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-10TeV_woAbs_smooth_10', 'C', "")
# chi2_pdf_plot(chi2,file_plot_dir+'Chi2andRint_pdf_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-10TeV_woAbs_smooth_10.png',10)
# plot_chi_squared(chi2, file_plot_dir, 'Chi2_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-12TeV_woAbs_smooth_10', 'C', "χ² reducido")
# plot_chi_squared(Rint, file_plot_dir, 'Rint_skymap_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-12TeV_woAbs_smooth_10', 'C', "")
# chi2_pdf_plot(chi2,file_plot_dir+'Chi2andRint_pdf_real-mapping_pwrind-2p6_all-wei_n16_ang5_dof10_gauss-12TeV_woAbs_smooth_10.png',10)

# outer radius energy distribution skymaps
# wion = load_from_npz(file_maps_dir+"wion_n16_ang5_dof10.npz", 'wion')
# plot_distribution(wion, file_plot_dir, 'wion_skymap_noWeights_n16_ang5_dof10_smooth2', 'C', 'Rango pesos')
wion = load_from_npz(file_maps_dir+"wion_allwei_n16_ang5_dof10.npz", 'wion')
plot_distribution(wion, file_plot_dir, 'wion_skymap_allWeights_n16_ang5_dof10', 'C', 'Rango pesos')
