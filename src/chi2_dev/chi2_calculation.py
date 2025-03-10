
'''
This script processes cosmic ray particle data to compute and analyze the Chi² distribution 
using skymaps with HEALPix, performing statistical tests and generating visualizations.

Inputs:
- Particle data file (.npz format) containing reweighted particles with energies and weights.
- Parameters for Chi² tests, including energy limits and skymap pixel width.
- Output directory for plots and results.

Outputs:
- Chi² values computed through statistical comparisons of skymap distributions.
- Plots of energy distributions, Chi² probability density functions, and skymaps.

Main Steps:
1. Load particle data from file.
2. Generate skymap distributions using ring and strip methods.
3. Perform Chi² tests on energy-weighted distributions.
4. Visualize and save the results as histograms and skymaps.
'''

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import pylab
import random
import scipy.stats 
from tqdm import tqdm
import gc

DEGREES_OF_FREEDOM = 20

# Improved data loading function
def load_data(file_path):
    """
    Loads data from a .npz file and returns it as an ndarray.
    This function simplifies the process by removing the redundant use of pickle.
    
    Parameters:
    - file_path: path to the .npz file containing the data.

    Returns:
    - ndarray containing the data content.
    """
    try:
        with np.load(file_path) as data:
            particles = data['reweighed_particles']  # Load directly from .npz
        return particles
    except IOError as e:
        print(f"Error loading .npz file: {e}")
        return None

# Probability density function for power law functions
def powerlaw_pdf(x, x_min, x_max, power):
    """
    Computes the probability density function (PDF) of a power-law distribution.

    Parameters:
    - x (float or array-like): The variable(s) at which to evaluate the PDF.
    - x_min (float): The minimum value of x in the distribution.
    - x_max (float): The maximum value of x in the distribution.
    - power (float): The exponent of the power law (typically negative for physical scenarios).

    Returns:
    - float or array-like: The value(s) of the PDF at x.
    
    The function is designed to handle the special case where the power is -1.0, avoiding 
    a division by zero and instead using the natural logarithm for normalization.
    """
    
    # Calculate the generalized limits for normalization
    x_min_g, x_max_g = x_min ** (power + 1.), x_max ** (power + 1.)
    
    # Special case for power = -1.0, avoids division by zero
    if power == -1.0:
        # Normalization using the logarithm of the ratio of limits
        return x ** power / np.log(x_max / x_min)
    else:
        # General formula for the power-law PDF normalization
        return (power + 1.) / (x_max_g - x_min_g) * x ** power


# Weighting scheme for energy bins
def weight_powerlaw(x, x_min, x_max, g, power):
    """
    Computes the weight for energy bins following a power-law distribution.
    
    Parameters:
    - x (float or array-like): The energy value(s) at which to evaluate the weight.
    - x_min (float): The minimum energy boundary of the distribution.
    - x_max (float): The maximum energy boundary of the distribution.
    - g (float): The weighting exponent applied to the energy.
    - power (float): The exponent of the power-law distribution.

    Returns:
    - float or array-like: The computed weight(s) for the input energy value(s).

    The function generates weights by scaling the energy values with an exponent 'g'
    and normalizing them by the power-law probability density function.
    """
    
    # Calculate the weight using a power-law function and normalize by the PDF
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)


def generate_log_uniform_data(low, high, sample_size):
    """
    Generates random data following a log-uniform (logarithmically uniform) distribution.
    
    Parameters:
    - low (float): The lower bound of the distribution (must be > 0).
    - high (float): The upper bound of the distribution.
    - sample_size (int): The number of random samples to generate.
    
    Returns:
    - ndarray: An array of random samples distributed logarithmically between low and high.
    
    This function is useful for scenarios where the data spans several orders of magnitude,
    ensuring a uniform distribution in the logarithmic scale.
    """
    
    # Check if the lower bound is valid (must be greater than 0)
    if low <= 0:
        raise ValueError("low limit must be higher than 0")
    
    # Convert the bounds to the logarithmic scale (base 10)
    log_low = np.log10(low)
    log_high = np.log10(high)
    
    # Generate uniform samples in the logarithmic space
    log_data = np.random.uniform(log_low, log_high, sample_size)
    
    # Transform the samples back to the original scale using 10^x
    data = np.power(10, log_data)
    
    return data

def generate_normal_weights(mean, std_dev, sample_size):
    """
    Generates normalized weights from a normal (Gaussian) distribution.
    
    Parameters:
    - mean (float): The mean of the normal distribution.
    - std_dev (float): The standard deviation of the normal distribution.
    - sample_size (int): The number of random weights to generate.
    
    Returns:
    - ndarray: An array of normalized weights ranging between 0 and 1.
    
    The function ensures the generated weights are scaled to a [0, 1] range, 
    which is useful for scenarios where normalized weights are required.
    """
    
    # Generate random weights from a normal distribution
    weights = np.random.normal(mean, std_dev, sample_size)
    
    # Normalize the weights to the range [0, 1] if possible
    if np.max(weights) > np.min(weights):
        normalized_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    else:
        # If all weights are the same, return the original array (avoids division by zero)
        normalized_weights = weights
    
    return normalized_weights


# Improved plot function with log-log scaling for energy and weight distributions
def plot_energy_histogram(energies, filename="energy_histogram.png"):
    """
    Plots a histogram of energy distribution and saves the figure.
    Filters near-zero values to avoid log(0) issues.
    
    Parameters:
    - energies: array with energy values.
    - filename: name of the file where the figure will be saved (default: 'energy_histogram.png').
    """
    energies = energies[energies > 0]  # Filter to avoid log(0)

    plt.figure(figsize=(6, 5))

    # Histogram for energies
    plt.hist(energies.flatten(), bins=DEGREES_OF_FREEDOM+1, log=True)
    plt.title('Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Frequency of events')

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')  
    plt.show()


# Chi² calculation with control for memory
def generic_shuffle_test(particles, limits, width, ndist):
    """
    Performs a Chi² test with a shuffle approach and adjusts energy limits.
    
    Parameters:
    - particles: array of particles (energies and weights).
    - limits: energy limits [min, max].
    - width: width to define neighboring pixels in the skymap.
    - ndist: number of distributions to process.
    
    Returns:
    - List of calculated Chi² values.
    """
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)

    # Adjusted limits for momentum
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))

    ext_boundary_dist = []
    for i in tqdm(range(ndist)):
        pixel_distribution, _ = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        ext_boundary_dist.append(pixel_distribution)

    # Shuffle the list directly instead of duplicating it
    np.random.shuffle(ext_boundary_dist)

    # Optimized Chi² calculation with memory release
    chi2_concat = []
    for i in range(ndist):
        data1 = ext_boundary_dist[i][0]
        wei1 = ext_boundary_dist[i][1]

        random_index = np.random.randint(ndist)
        data2 = ext_boundary_dist[random_index][0]
        wei2 = ext_boundary_dist[random_index][1]

        chi2 = test_weights_v3(data1, data2, wei1, wei2)
        chi2_concat.append(chi2)

    del ext_boundary_dist  # Free memory immediately after use
    gc.collect()  # Explicit memory release

    return chi2_concat

# Utility functions for skymap data manipulation
def get_pixel_distribution(pixel):
    """
    Extracts energy and weight data from a single pixel in the skymap.
    
    Parameters:
    - pixel (list of lists): A list containing particle data for a specific pixel. 
      Each element in the list is a [energy, weight] pair.
    
    Returns:
    - ndarray: A 2D array where the first row contains energies and the second row contains weights.
    
    Example:
    Input: pixel = [[1.2, 0.5], [2.3, 0.8]]
    Output: array([[1.2, 2.3], [0.5, 0.8]])
    """
    
    # Extract the energy values from the first element of each particle
    energies = [particle[0] for particle in pixel]
    
    # Extract the weight values from the second element of each particle
    weights = [particle[1] for particle in pixel]
    
    # Return as a 2D numpy array with energies and weights in separate rows
    return np.array([energies, weights])


def get_sky_distribution(pixel_list):
    """
    Aggregates energy and weight data from all pixels in a skymap.
    
    Parameters:
    - pixel_list (list of lists): A list where each element is a pixel, 
      and each pixel is a list of [energy, weight] pairs.
    
    Returns:
    - ndarray: A 2D array with all energies in the first row and all weights in the second row.
    
    Example:
    Input: pixel_list = [[[1.2, 0.5], [2.3, 0.8]], [[1.5, 0.6], [2.8, 0.9]]]
    Output: array([[1.2, 2.3, 1.5, 2.8], [0.5, 0.8, 0.6, 0.9]])
    """
    
    # Initialize empty lists to accumulate energy and weight data
    energies = []
    weights = []
    
    # Iterate over each pixel in the skymap
    for pixel in pixel_list:
        # Use get_pixel_distribution to extract data from the current pixel
        distribution = get_pixel_distribution(pixel)
        
        # Append energies and weights to the main lists
        energies += distribution[0].tolist()
        weights += distribution[1].tolist()
    
    # Return as a 2D numpy array with energies and weights in separate rows
    return np.array([energies, weights])


def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    """
    Computes the distribution of energies and weights in a strip around a given pixel
    on a HEALPix skymap, excluding a ring directly surrounding the pixel.
    
    Parameters:
    - pixel_number (int): Index of the central pixel in the skymap.
    - pixel_list (list of lists): Skymap data where each element is a pixel containing 
      a list of [energy, weight] pairs.
    - nside (int): The HEALPix Nside parameter, which determines the resolution of the skymap.
    - num_pixels (int): The width of the strip in pixel units.

    Returns:
    - ndarray: A 2D array where the first row contains energies and the second row contains weights.
    - int: The total number of pixels included in the strip.
    
    The function uses HEALPix utilities to define a strip of pixels in a latitude band 
    while excluding a ring around the central pixel.
    """
    
    # Convert the pixel index to spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, pixel_number)
    
    # Get the 3D vector for the pixel direction
    vec = hp.pix2vec(nside, pixel_number)
    
    # Compute the angular size of a pixel (in radians)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    
    # Get all pixels in a strip of latitude around the central pixel
    strip = hp.query_strip(nside, 
                           theta - num_pixels * d_theta, 
                           theta + num_pixels * d_theta)
    
    # Define a ring around the central pixel with the same width
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    
    # Exclude the ring from the strip using set difference
    strip = np.setdiff1d(strip, particle_ring)
    
    # Get the energy and weight distribution of the remaining pixels in the strip
    return get_sky_distribution(pixel_list[strip]), len(strip)


def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    """
    Computes the distribution of energies and weights in a circular ring around a given pixel
    on a HEALPix skymap.
    
    Parameters:
    - pixel_number (int): Index of the central pixel in the skymap.
    - pixel_list (list of lists): Skymap data where each element is a pixel containing 
      a list of [energy, weight] pairs.
    - nside (int): The HEALPix Nside parameter, which determines the resolution of the skymap.
    - num_pixels (int): The radius of the ring in pixel units.
    
    Returns:
    - ndarray: A 2D array where the first row contains energies and the second row contains weights.
    - int: The total number of pixels included in the ring.
    
    The function uses HEALPix utilities to define a circular ring of pixels surrounding the central pixel.
    """
    
    # Get the 3D vector for the central pixel direction
    vec = hp.pix2vec(nside, pixel_number)
    
    # Calculate the angular size of a pixel in radians
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    
    # Generate a circular ring of pixels around the central pixel
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    
    # Get the energy and weight distribution of the pixels in the ring
    return get_sky_distribution(pixel_list[particle_ring]), len(particle_ring)


def impose_energy_range(distribution, min_energy, max_energy):
    """
    Filters a distribution of energies and weights to impose a specific energy range.
    
    Parameters:
    - distribution (ndarray): A 2D array where the first row contains energy values 
      and the second row contains corresponding weights.
    - min_energy (float): The minimum energy threshold for filtering.
    - max_energy (float): The maximum energy threshold for filtering.
    
    Returns:
    - ndarray: A 2D array with energies and weights only within the specified energy range.
    
    This function is used to constrain data to a relevant energy range, which is particularly 
    important for statistical analyses and avoiding outlier effects.
    """
    
    # Extract energies and weights from the input distribution
    energies = distribution[0]
    weights = distribution[1]
    
    # Identify indices where the energies fall within the specified range
    indices = np.where(np.logical_and(energies >= min_energy, energies <= max_energy))
    
    # Filter and return the energies and weights using the identified indices
    return np.array([energies[indices], weights[indices]])

# Chi² PDF plot function
def chi2_pdf_plot(chi2_concat):
    """
    Plots a Probability Density Function (PDF) of Chi² values and compares it 
    to the theoretical Chi² distribution.

    Parameters:
    - chi2_concat (list or array-like): List of calculated Chi² values.

    The function generates a log-log plot comparing the empirical distribution of
    Chi² values to the expected theoretical Chi² PDF with the specified degrees of freedom.
    """

    # Define logarithmically spaced bins for the histogram
    xbins = np.logspace(-2, 3, 100)

    # Calculate the histogram of the Chi² values, normalized to form a probability density
    hist, edges = np.histogram(chi2_concat, bins=xbins, density=True)

    # Create a theoretical Chi² distribution with a defined number of degrees of freedom
    rv = scipy.stats.chi2(DEGREES_OF_FREEDOM)

    # Initialize the plot with a defined size
    plt.figure(figsize=(10, 6))

    # Plot the theoretical Chi² probability density function
    plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, 
             label=f'pdf ({DEGREES_OF_FREEDOM} dof)')

    # Plot the empirical histogram of the Chi² values
    plt.plot(edges[1:], hist, label='data')

    # Set logarithmic scales for both axes
    plt.yscale('log')
    plt.xscale('log')

    # Add plot titles and labels
    plt.title('Chi² Probability Density Function (PDF)')
    plt.xlabel('Chi² sum')
    plt.ylabel('Density')

    # Display the legend and grid
    plt.legend()
    plt.grid(True)

    # Render the plot
    plt.show()


# perform_test_weights_v3 function
def perform_test_weights_v3(particles, limits, width):
    """
    Computes Chi² for all pixels in the skymap using strip and ring distributions.
    
    Parameters:
    - particles: 3D array of particles in each skymap pixel.
    - limits: energy limits [min, max].
    - width: width for defining strip and ring areas.
    
    Returns:
    - List of calculated Chi² values for each pixel.
    """
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)

    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))

    chi2sum = []
    for i in tqdm(range(npix)):
        # print('PIXEL', i)
        strip_distribution, _ = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution, _ = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        chi2 = test_weights_v3(pixel_distribution[0], strip_distribution[0],
                               pixel_distribution[1], strip_distribution[1])
        chi2sum.append(chi2)
    return chi2sum

# Test and plot functions
def test_weights_v3(data1, data2, wei1, wei2):
    """
    Computes the Chi² statistic for comparing two weighted histograms of energy distributions.
    
    Parameters:
    - data1, data2 (array-like): Arrays of energy values from two distributions to compare.
    - wei1, wei2 (array-like): Corresponding weights for data1 and data2.
    
    Returns:
    - float: The Chi² sum or an alternative metric based on weighted histogram comparison.
    
    This function performs a weighted Chi² test by comparing histograms of the two data sets.
    It uses logarithmically spaced bins and normalizes the weights before computing the statistic.
    """
    
    # Determine the minimum and maximum values for the histogram bins
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    
    # Define the number of bins (degrees of freedom + 1)
    bins_count = DEGREES_OF_FREEDOM + 1

    # Generate logarithmically spaced bins for the histograms
    ebins = np.logspace(np.log10(min_val), np.log10(max_val), bins_count)

    # Normalize weights to ensure sum equals 1
    norm1 = np.sum(wei1)
    norm2 = np.sum(wei2)
    wei1_norm = wei1 / norm1
    wei2_norm = wei2 / norm2

    # Create histograms for unweighted data to identify valid bins
    N_on, edges = np.histogram(data1, bins=ebins)
    N_off, _ = np.histogram(data2, bins=ebins)
    
    # Check if any bin has less than 20 entries (considered statistically insufficient)
    if (N_on < 20).any() or (N_off < 20).any():
        print("CONDITION NOT MET")
    
    # Create a mask to select only valid bins with enough data
    mask = np.logical_and(N_on > 20, N_off > 20)

    # Weighted histograms and their squared weights
    Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1_norm)
    S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1_norm, 2))
    Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2_norm)
    S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2_norm, 2))

    # Identify bins with valid weights
    valid_Wi_on = Wi_on > 0
    valid_Wi_off = Wi_off > 0

    # Initialize the variance array for the weighted histograms
    di2 = np.zeros_like(Wi_off, dtype=np.float64)
    
    # Calculate the variance for valid bins
    di2[valid_Wi_on & valid_Wi_off] = Wi_off[valid_Wi_on & valid_Wi_off] * (
        S2i_on[valid_Wi_on & valid_Wi_off] / Wi_on[valid_Wi_on & valid_Wi_off] + 
        S2i_off[valid_Wi_on & valid_Wi_off] / Wi_off[valid_Wi_on & valid_Wi_off])
    
    # Set variance to infinity where weights are not valid
    di2[~(valid_Wi_on & valid_Wi_off)] = np.inf

    # Calculate the Chi² statistic only for the valid bins
    chi2sum = np.sum(np.power((Wi_on[mask] - Wi_off[mask]), 2) / di2[mask])
    
    # Optionally plot histograms if Chi² sum is below a threshold
    if chi2sum <= 40.0:
        print('CHI2SUM', chi2sum)
        fig_name1 ='/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/energy_hist_Wion_chi2-.png'
        fig_name2 ='/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/energy_hist_Wioff_chi2-.png'
        plot_energy_histogram(data1, fig_name1)
        plot_energy_histogram(data2, fig_name2)
    
    # Alternative return: Chi² sum or total weight in valid bins
    return np.sum(Wi_on[mask])


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
                dMin=chi2sum[np.argmin(chi2sum)],
                dMax=chi2sum[np.argmax(chi2sum)],
                filename=out_dir + name)
    
    # Close the plot to free memory
    plt.close()


# For rotating sky maps to equatorial coordinates
def rotate_map(old_map):
    """
    Rotates a skymap from its original coordinate system to equatorial coordinates.
    
    Parameters:
    - old_map (ndarray): 1D array representing the pixel values of the input skymap.
    
    Returns:
    - new_map (ndarray): 1D array of the same size as old_map, representing the skymap
      rotated to equatorial coordinates.
    
    The function uses a series of matrix transformations and the HEALPix Rotator to 
    convert pixel values to the new coordinate system.
    """
    
    # Coordinate transformation matrix from simulation to ecliptic coordinates
    coord_matrix = np.matrix([
        [-0.202372670869508942, 0.971639226673224665, 0.122321361599999998],
        [-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],
        [-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]
    ])

    # Transformation matrix to align GMT on the right-hand side of maps
    map_matrix = np.matrix([
        [-1, 0, 0], 
        [0, -1, 0], 
        [0, 0, 1]
    ])
    
    # Determine the number of pixels and HEALPix nside parameter
    npix = len(old_map)
    nside = hp.npix2nside(npix)
    
    # Initialize the new map with zeros
    new_map = np.zeros(npix)
    
    # Create a Rotator to handle the transition from celestial to equatorial coordinates
    r = hp.Rotator(coord=['C', 'E'])

    # Iterate over all pixels in the new map
    for i in range(npix):
        # Convert pixel index to spherical coordinates (theta, phi)
        theta, phi = hp.pix2ang(nside, i)

        # Transform from simulation to ecliptic coordinates
        old_theta, old_phi = hp.rotator.rotateDirection(
            np.linalg.inv(map_matrix), theta, phi)

        # Apply transformation to equatorial coordinates using the Rotator
        old_theta, old_phi = r(old_theta, old_phi)

        # Apply the final transformation to align GMT correctly
        old_theta, old_phi = hp.rotator.rotateDirection(
            np.linalg.inv(coord_matrix), old_theta, old_phi)

        # Map the transformed coordinates back to a pixel index
        old_pix = hp.ang2pix(nside, old_theta, old_phi)

        # Assign the value from the old map to the new map
        new_map[i] += old_map[old_pix]

    return new_map


# Example usage of the improved functions

# Define the path to the particle data file (.npz format)
particles_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/rewei_part_nside=16_dof=120_pwrind=-1_real-mapping.npz'

# Load the particle data using the improved load_data function
particles = load_data(particles_dir)

# Define the output directory for plots and results
file_plot_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/'

# Check if particle data was loaded successfully
if particles is not None:
    
    # Separate the loaded data into energy and weight arrays
    energies = particles[:, :, 0]  # Extract energies from the first element of each pair
    weights = particles[:, :, 1]   # Extract weights from the second element of each pair
    
    # Optionally generate and save histograms of energies and weights
    # plot_energy_weight_histograms(energies, weights)
    
    # Perform the Chi² test using the perform_test_weights_v3 function
    # The test uses an energy range of [0.1, 100] and a strip width of 4 pixels
    chi2_result = perform_test_weights_v3(particles, [0.1, 100], 4)
    
    # Rotate the Chi² map to equatorial coordinates
    chi2_result = rotate_map(chi2_result)
    
    # Save the rotated Chi² map to a compressed .npz file (optional)
    # np.savez_compressed('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/' + 
    #                     'Wion-sum_nside=16_bins=120_pwrind=-1_pix=1_real-mapping_energy-5-30' + 
    #                     ".npz", chi2=chi2_result)
    
    # Plot the Chi² skymap and save the image to the specified directory
    plot_chi_squared(chi2_result, file_plot_dir, 
                     'skymap_chi2_nside=16_bins=120_pwrind=-1_pix=4_real-mapping_energy-0p1-100')
    
    # Generate and display the Chi² Probability Density Function (PDF) plot
    chi2_pdf_plot(chi2_result)

else:
    print("Data loading failed.")


