import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import pylab
import random
import scipy.stats 
from tqdm import tqdm
import gc
import pickle

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

def hist_plot(edges, Wi_on, Wi_off, S2i_on, S2i_off, di2):
    plt.figure(figsize=(10, 6))
    # plt.errorbar(edges[1:], Wi_on, yerr=np.sqrt(S2i_on))
    # plt.errorbar(edges[1:], Wi_off, yerr=np.sqrt(S2i_off))
    # plt.errorbar(edges[1:], np.abs(Wi_on - Wi_off), yerr=np.sqrt(S2i_on))
    # plt.plot(edges[1:], np.abs(Wi_on - Wi_off), label = 'Wi_on - Wi_off')
    plt.plot(edges[1:], np.power((Wi_on - Wi_off), 2), label = '(Wi_on - Wi_off)^2')
    # plt.plot(edges[1:], np.sqrt(S2i_on), label = 'S2i_on')
    # plt.plot(edges[1:], np.sqrt(S2i_off), label = 'S2i_off')
    plt.plot(edges[1:], di2, label = 'di2')
    plt.title('Error Wi_on vs Wi_off')
    plt.xlabel('Energy bins')
    plt.ylabel('Weighted counts')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def chi2_pdf_plot(chi2_concat):
    xbins = np.logspace(-2, 3, 100)
    hist, edges = np.histogram(chi2_concat, bins=xbins, density=True)
    rv = scipy.stats.chi2(20)
    plt.figure(figsize=(10, 6))
    plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, label='pdf (20 dof)')
    plt.plot(edges[1:], hist, label = 'data')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Chi2 PDF')
    plt.xlabel('Chi2sum')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()    

def test_weights_v3(data1, data2, wei1, wei2):
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    bins_count = 21

    if min_val <= 0:
        min_val = 1e-10  # Small positive value

    ebins = np.logspace(np.log10(min_val), np.log10(max_val), bins_count)

    # Normalize weights
    norm1 = np.sum(wei1)
    norm2 = np.sum(wei2)
    wei1_norm = wei1 / norm1
    wei2_norm = wei2 / norm2

    # Histogram
    N_on, edges = np.histogram(data1, bins=ebins)
    N_off, _ = np.histogram(data2, bins=ebins)
    mask = np.logical_and(N_on>20, N_off>20)

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
    chi2sum_red = chi2sum / (len(ebins) - 1)
    # Plots on/off histograms per pixel
    # hist_plot(edges, Wi_on, Wi_off, S2i_on, S2i_off)
    # if chi2sum > 90.0:
    #     hist_plot(edges, Wi_on, Wi_off, S2i_on, S2i_off, di2)
    return chi2sum

def generic_chi2_test(sample_size, low, high):
    # Generate an array with random normal distributions at the boundary layer
    ext_boundary_dist = []
    for _ in range(sample_size):
        energies = generate_log_uniform_data(low, high, sample_size)
        weights = weight_powerlaw(energies, low, high, -2.6, -1)
        ext_boundary_dist.append([energies, weights])

    # Randomly map the boundary distribution array to the inner boundary
    int_boundary_dist = ext_boundary_dist.copy()

    random.shuffle(int_boundary_dist)

    # Calculate chi2
    chi2_concat = []
    for i in range(sample_size):
        data1 = int_boundary_dist[i][0]  # energy values from shuffled list
        wei1 = int_boundary_dist[i][1]  # weight values from shuffled list

        # Select a random index for data2 and wei2
        random_index = np.random.randint(sample_size)
        data2 = int_boundary_dist[random_index][0]  # energy values at random index
        wei2 = int_boundary_dist[random_index][1]  # corresponding weight values at random index

        chi2 = test_weights_v3(data1, data2, wei1, wei2)
        chi2_concat.append(chi2)
    return chi2_concat

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&CHI2 GENERIC TEST&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
# Global parameters
sample_size = 5000

# Parameters for Uniform data
low = 5.287760354365922
high = 14.985871050425096

# Parameters for NOrmal data
mean3 = 1.0
std_dev3 = 0.5
mean4 = 1.0
std_dev4 = 0.5

# Plot chi2 calculation vs chi2 pdf
# chi2_pdf_plot(generic_chi2_test(sample_size, low, high))
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

# PARTICLE TESTING #

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
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

def perform_test_weights_v3(particles, limits, width):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
    #print('energy limits lower/upper',lower, upper)

    chi2sum = []
    #for i in tqdm(range(256,npix)):
    for i in tqdm(range(npix)):
        strip_distribution, _ = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution, _ = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        chi2 = test_weights_v3(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1])
        chi2sum.append(chi2)
    return chi2sum

def generic_shuffle_test(particles, limits, width, ndist):
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))

    ext_boundary_dist = []

    for i in tqdm(range(ndist)):
        pixel_distribution, _ = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        ext_boundary_dist.append(pixel_distribution)

    # Shuffle the list directly
    np.random.shuffle(ext_boundary_dist)

    # Calculate chi2
    chi2_concat = []
    for i in range(ndist):
        data1 = ext_boundary_dist[i][0]
        wei1 = ext_boundary_dist[i][1]

        random_index = np.random.randint(ndist)
        data2 = ext_boundary_dist[random_index][0]
        wei2 = ext_boundary_dist[random_index][1]

        chi2 = test_weights_v3(data1, data2, wei1, wei2)
        chi2_concat.append(chi2)

    del ext_boundary_dist
    gc.collect()

    return chi2_concat

def plot_skymap(skymap, title, proj='C', label='', filename=None, thresh=None,
                dMin=None, dMax=None, sun=None):
    params = {'legend.fontsize': 'x-large',
              # 'axes.labelsize': 'x-large',
              'axes.titlesize': '20',
              # 'xtick.labelsize': 'x-large',
              # 'ytick.labelsize': 'x-large',
              "font.family": "serif"}
    pylab.rcParams.update(params)

    colormap = pylab.get_cmap("coolwarm")
    colormap.set_under("w")

    if proj == 'C0':
        rotation = (0, 0, 0)
    elif proj == 'C':
        rotation = (-180, 0, 0)
    else:
        rotation = proj

    hp.mollview(skymap,
                fig=1,
                title=title,
                rot=rotation,  # coord=['C'],
                unit=label,
                margins=(0.0, 0.03, 0.0, 0.13),
                notext=False, cmap=colormap, min=dMin, max=dMax)

    fig = pylab.figure(1)
    for ax in fig.get_axes()[0:1]:
        if proj == 'C0':
            ax.annotate(r"0$^\circ$", xy=(1.8, 0.625), size="x-large")
            ax.annotate(r"360$^\circ$", xy=(-1.95, 0.625), size="x-large")
        elif proj == 'C':
            ax.annotate(r"0$^\circ$", xy=(1.8, 0.625), size="x-large")
            ax.annotate(r"360$^\circ$", xy=(-1.95, 0.625), size="x-large")

    if sun is not None:
        hp.projscatter(sun[0], sun[1], lonlat=True, coord='C')
        hp.projtext(sun[0], sun[1], 'Sun', lonlat=True, coord='C', fontsize=18)

    hp.graticule()

    if filename:
        fig.savefig(filename, dpi=250)

def plot_chi_squared(chi2sum, out_dir, name):
    print('chi2sum[np.argmax(chi2sum)] ' + name, chi2sum[np.argmax(chi2sum)])
    print('chi2sum[np.argmin(chi2sum)] ' + name, chi2sum[np.argmin(chi2sum)])
    #signs = np.sign(p_values)
    z_values = np.abs(chi2sum)
    #z_values = np.maximum(-stat.norm.ppf(np.abs(p_values)), 0) * signs
    #z_values = np.maximum(-stat.norm.ppf(np.abs(np.nan_to_num(p_values))), 0) * np.nan_to_num(signs)
    plot_skymap(z_values,
                title=name,
                label="Range",
                proj='C0',
                dMin=0.0,
                dMax=5.0,
                filename=out_dir + name)
    plt.close()

def load_npz_file_as_ndarray(file_path):
    try:
        # Load the .npz file
        with np.load(file_path) as data:
            # Convert to a list of arrays
            arrays = [data[key] for key in data.files]
            # Stack arrays into a single NDArray
            particles = np.vstack(arrays)
        return particles

    except IOError as e:
        print(f"Error loading .npz file: {e}")
        return None

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TEST PARTICLE DATA&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
particles_dir = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/phyindex_2p6_nside8_rand-finalpix.npz"
with open(particles_dir, 'rb') as f:
    particles = pickle.load(f)

particles = load_npz_file_as_ndarray(particles_dir)
figs_dir = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/chi2_debug/"
maps_dir = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/maps/"

chi2_test = perform_test_weights_v3(particles, [0.1, 100], 5)
#np.savez_compressed(maps_dir + "chi2_2p6_nside8_rand-finalpix" + ".npz", data=chi2_test)
chi2_pdf_plot(chi2_test)
# plot_chi_squared(chi2_test, figs_dir, "chi2_skymap_1")
# chi2_test2 = generic_shuffle_test(particles, [0.1, 100], 5, 10)
# chi2_pdf_plot(chi2_test2)
