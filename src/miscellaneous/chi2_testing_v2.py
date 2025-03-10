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
    x_min_g, x_max_g = x_min ** (power + 1.), x_max ** (power + 1.)
    if power == -1.0:
        return x ** power / np.log(x_max / x_min)
    else:
        return (power + 1.) / (x_max_g - x_min_g) * x ** power

# Weighting scheme for energy bins
def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)

def generate_log_uniform_data(low, high, sample_size):
    if low <= 0:
        raise ValueError("low limit must be higher than 0")
    log_low = np.log10(low)
    log_high = np.log10(high)
    log_data = np.random.uniform(log_low, log_high, sample_size)
    data = np.power(10, log_data)
    return data

def generate_normal_weights(mean, std_dev, sample_size):
    weights = np.random.normal(mean, std_dev, sample_size)
    return (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) > np.min(weights) else weights

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
    energies = [particle[0] for particle in pixel]
    weights = [particle[1] for particle in pixel]
    return np.array([energies, weights])

def get_sky_distribution(pixel_list):
    energies = []
    weights = []
    for pixel in pixel_list:
        distribution = get_pixel_distribution(pixel)
        energies += distribution[0].tolist()
        weights += distribution[1].tolist()
    return np.array([energies, weights])

def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    theta, phi = hp.pix2ang(nside, pixel_number)
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    strip = hp.query_strip(nside, theta - num_pixels * d_theta, theta + num_pixels * d_theta)
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    strip = np.setdiff1d(strip, particle_ring)
    return get_sky_distribution(pixel_list[strip]), len(strip)

def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    return get_sky_distribution(pixel_list[particle_ring]), len(particle_ring)

def impose_energy_range(distribution, min_energy, max_energy):
    energies = distribution[0]
    weights = distribution[1]
    indices = np.where(np.logical_and(energies >= min_energy, energies <= max_energy))
    return np.array([energies[indices], weights[indices]])

# Chi² PDF plot function
def chi2_pdf_plot(chi2_concat):
    """
    Plots a PDF of Chi² values and compares it to the theoretical Chi² distribution.
    
    Parameters:
    - chi2_concat: list of calculated Chi² values.
    """
    xbins = np.logspace(-2, 3, 100)
    hist, edges = np.histogram(chi2_concat, bins=xbins, density=True)
    rv = scipy.stats.chi2(DEGREES_OF_FREEDOM)  # Chi² with N degrees of freedom
    
    plt.figure(figsize=(10, 6))
    plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, label='pdf ('+ str(DEGREES_OF_FREEDOM) + ' dof)')
    plt.plot(edges[1:], hist, label = 'data')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Chi² PDF')
    plt.xlabel('Chi² sum')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
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
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    bins_count = DEGREES_OF_FREEDOM + 1

    # if min_val <= 0:
    #     min_val = 1e-10  # Small positive value

    ebins = np.logspace(np.log10(min_val), np.log10(max_val), bins_count)

    # Normalize weights
    norm1 = np.sum(wei1)
    norm2 = np.sum(wei2)
    wei1_norm = wei1 / norm1
    wei2_norm = wei2 / norm2

    # Histogram
    N_on, edges = np.histogram(data1, bins=ebins)
    N_off, _ = np.histogram(data2, bins=ebins)
    if (N_on < 20).any() or (N_off < 20).any():
        print("CONDITION NOT MET")
    mask = np.logical_and(N_on > 20, N_off > 20)

    Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1_norm)
    S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1_norm, 2))
    Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2_norm)
    S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2_norm, 2))

    valid_Wi_on = Wi_on > 0
    valid_Wi_off = Wi_off > 0
    di2 = np.zeros_like(Wi_off, dtype=np.float64)
    di2[valid_Wi_on & valid_Wi_off] = Wi_off[valid_Wi_on & valid_Wi_off] * (
        S2i_on[valid_Wi_on & valid_Wi_off] / Wi_on[valid_Wi_on & valid_Wi_off] + 
        S2i_off[valid_Wi_on & valid_Wi_off] / Wi_off[valid_Wi_on & valid_Wi_off])
    di2[~(valid_Wi_on & valid_Wi_off)] = np.inf

    # Chi2 calculation
    chi2sum = np.sum(np.power((Wi_on[mask] - Wi_off[mask]), 2) / di2[mask])
    if chi2sum <= 40.0:
        print('CHI2SUM', chi2sum)
        fig_name1 ='/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/energy_hist_Wion_chi2-.png'
        fig_name2 ='/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/energy_hist_Wioff_chi2-.png'
        plot_energy_histogram(data1, fig_name1)
        plot_energy_histogram(data2, fig_name2)
    # return chi2sum
    return np.sum(Wi_on[mask])

def plot_skymap(skymap, title, proj='C', label='', filename=None, thresh=None,
                dMin=None, dMax=None, sun=None):
    # Configure plot parameters
    params = {'legend.fontsize': 'x-large',
              'axes.titlesize': '20',
              "font.family": "serif"}
    pylab.rcParams.update(params)

    # Define colormap
    colormap = plt.get_cmap("coolwarm")
    
    # Check if set_under is necessary
    if hasattr(colormap, 'set_under'):
        colormap.set_under("w")

    # Set rotation based on projection type
    rotation = (0, 0, 0) if proj == 'C0' else (-180, 0, 0) if proj == 'C' else proj

    # Plot the sky map using Mollweide projection
    hp.mollview(skymap, title=title, rot=rotation, unit=label,
                margins=(0.0, 0.03, 0.0, 0.13), notext=False,
                cmap=colormap, min=dMin, max=dMax)
    
    # Get the current figure
    fig = plt.gcf()
    ax = fig.get_axes()[0] if fig.get_axes() else None
    
    # Add annotations for 0° and 360° if applicable
    if ax and proj in ['C', 'C0']:
        ax.annotate(r"0$^\circ$", xy=(1.8, 0.625), size="x-large")
        ax.annotate(r"360$^\circ$", xy=(-1.95, 0.625), size="x-large")
    
    # Mark the Sun's position if provided
    if sun is not None:
        hp.projscatter(sun[0], sun[1], lonlat=True, coord='C')
        hp.projtext(sun[0], sun[1], 'Sun', lonlat=True, coord='C', fontsize=18)
    
    # Add a grid overlay
    hp.graticule()

    # Save the figure if a filename is provided
    if filename:
        fig.savefig(filename, dpi=250)
        plt.close(fig)  # Free memory after saving


def plot_chi_squared(chi_squared_map, out_dir, name):
    chi2sum = chi_squared_map
    print('chi2sum[np.argmax(chi2sum)] ' + name, chi2sum[np.argmax(chi2sum)])
    print('chi2sum[np.argmin(chi2sum)] ' + name, chi2sum[np.argmin(chi2sum)])
    #print('chi2sum',chi2sum)
    #signs = np.sign(p_values)
    #print('signs',signs)
    z_values = np.abs(chi2sum)
    #z_values = np.maximum(-stat.norm.ppf(np.abs(p_values)), 0) * signs
    #z_values = np.maximum(-stat.norm.ppf(np.abs(np.nan_to_num(p_values))), 0) * np.nan_to_num(signs)
    plot_skymap(z_values,
                title=name,
                label="Range",
                proj='C0',
                dMin=chi2sum[np.argmin(chi2sum)],
                dMax=chi2sum[np.argmax(chi2sum)],
                filename=out_dir + name)
    plt.close()

# For rotating sky maps to equatorial coordinates
def rotate_map(old_map):
    coord_matrix = np.matrix([
        [-0.202372670869508942, 0.971639226673224665, 0.122321361599999998],
        [-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],
        [-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]
    ])

    map_matrix = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    npix = len(old_map)
    nside = hp.npix2nside(npix)
    new_map = np.zeros(npix)
    r = hp.Rotator(coord=['C', 'E'])

    # For each pixel in the new map, add the transformed pixel from the old map
    for i in range(npix):
        theta, phi = hp.pix2ang(nside, i)

        # Apply transform from simulation to ecliptic coordinates
        old_theta, old_phi = hp.rotator.rotateDirection(
            np.linalg.inv(map_matrix), theta, phi)

        # Appy transform from ecliptic to equatorial coordinates
        old_theta, old_phi = r(old_theta, old_phi)

        # Apply transform to put GMT on rhs of maps
        old_theta, old_phi = hp.rotator.rotateDirection(
            np.linalg.inv(coord_matrix), old_theta, old_phi)

        # Add appropriate pixel to new map
        old_pix = hp.ang2pix(nside, old_theta, old_phi)
        new_map[i] += old_map[old_pix]

    return new_map

# Example usage of the improved functions
particles_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/rewei_part_nside=16_dof=120_pwrind=-1_real-mapping.npz'
particles = load_data(particles_dir)
file_plot_dir = '/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/'

if particles is not None:
    # Separate energies and weights for histograms
    energies = particles[:, :, 0]
    weights = particles[:, :, 1]
    #plot_energy_weight_histograms(energies, weights)

    # Perform Chi² test and plot results
    chi2_result = perform_test_weights_v3(particles, [0.1, 100], 4)
    chi2_result = rotate_map(chi2_result)
    #np.savez_compressed('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/' + 'Wion-sum_nside=16_bins=120_pwrind=-1_pix=1_real-mapping_energy-5-30' + ".npz", chi2=chi2_result)
    plot_chi_squared(chi2_result, file_plot_dir, 'skymap_chi2_nside=16_bins=120_pwrind=-1_pix=4_real-mapping_energy-0p1-100')
    chi2_pdf_plot(chi2_result)
else:
    print("Data loading failed.")

