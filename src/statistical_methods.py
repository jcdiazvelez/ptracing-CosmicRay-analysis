import numpy as np
import healpy as hp
import scipy.stats as stat
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import pylab


# kolmogorov-p-Smirnov test for two weighted distributions
def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    combined = np.concatenate([data1, data2])
    c_wei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    c_wei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = c_wei1[np.searchsorted(data1, combined, side='right')]
    cdf2we = c_wei2[np.searchsorted(data2, combined, side='right')]
    diff = cdf1we - cdf2we
    abs_diff = np.abs(diff)
    d = np.max(abs_diff)
    ind = np.argmax(abs_diff)
    signed_d = diff[ind]
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = np.sqrt(m * n / (m + n))
    if alternative == 'two-sided':
        prob = stat.distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the largest of (n1, n2)
        expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)
    return signed_d, prob

# Chi-squared test for two weighted distributions
def chi_squared_weighted(data1, data2, wei1, wei2, npix_on, npix_off):
    # Create the energy histogram
    ebins = np.logspace(2.5,5,20)
    ehist_on, edges = np.histogram(data1,bins=ebins,weights=wei1)
    var_on, edges = np.histogram(data1,bins=ebins,weights=np.power(wei1,2))
    sigma_on = np.sqrt(var_on)
    ehist_off,edges = np.histogram(data2,bins=ebins,weights=wei2)
    var_off, edges = np.histogram(data2,bins=ebins,weights=np.power(wei2,2))
    sigma_off = np.sqrt(var_off)
    # Check for zero or very small denominators
    # denominator_on = np.where(sigma_on / npix_on == 0, np.inf, sigma_on / npix_on)
    # denominator_off = np.where(sigma_off / npix_off == 0, np.inf, sigma_off / npix_off)
    denominator_on = np.where(sigma_on / np.sum(ehist_on) == 0, np.inf, sigma_on / np.sum(ehist_on))
    denominator_off = np.where(sigma_off / np.sum(ehist_off) == 0, np.inf, sigma_off / np.sum(ehist_off))
    # Calculate chi-squared sum with handled denominators
    chi2sum = 0
    # chi2sum = np.sum(np.power((ehist_on/npix_on - ehist_off/npix_off), 2) / (np.power(denominator_on, 2) + np.power(denominator_off, 2)))
    chi2sum = np.sum(np.power((ehist_on/np.sum(ehist_on) - ehist_off/np.sum(ehist_off)), 2) / (np.power(denominator_on, 2) + np.power(denominator_off, 2)))
    chi2sum_red = chi2sum/(len(ebins)-1)

    # if chi2sum_red > 1600.0:
    #     plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')
    #     plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')
    #     plt.loglog()
    #     plt.legend()
    #     plt.xlabel("Energy bins")
    #     plt.ylabel("ehist/np.sum(ehist)")
    #     #plt.show()
    #     fig = pylab.figure(1)
    #     fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/chi_squared_april2/chi_squared_'+str(int(chi2sum_red))+'.png', dpi=250)
    #     plt.close()

    # if 800.0 < chi2sum_red < 850.0:
    #     plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')
    #     plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')
    #     plt.loglog()
    #     plt.legend()
    #     plt.xlabel("Energy bins")
    #     plt.ylabel("ehist/np.sum(ehist)")
    #     fig = pylab.figure(1)
    #     fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/chi_squared_april3/chi_squared_'+str(int(chi2sum_red))+'.png', dpi=250)
    #     plt.close()

    # if 110.0 < chi2sum_red < 120.0:
    #     plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')
    #     plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')
    #     plt.loglog()
    #     plt.legend()
    #     plt.xlabel("Energy bins")
    #     plt.ylabel("ehist/np.sum(ehist)")
    #     fig = pylab.figure(1)
    #     fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/chi_squared_april3/chi_squared_'+str(int(chi2sum_red))+'.png', dpi=250)
    #     plt.close()

    # if 0.0 < chi2sum_red < 5.0:
    #     plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')
    #     plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')
    #     plt.loglog()
    #     plt.legend()
    #     plt.xlabel("Energy bins")
    #     plt.ylabel("ehist/np.sum(ehist)")
    #     fig = pylab.figure(1)
    #     fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/chi_squared_april3/chi_squared_'+str(int(chi2sum_red))+'.png', dpi=250)
    #     plt.close()

    # # Signed diff
    # diff = ehist_on / npix_on - ehist_off / npix_off
    # print('diff', diff)
    # abs_diff = np.abs(diff)
    # ind = np.argmax(abs_diff)
    # signed_d = diff[ind]   
    # print('signed_d', signed_d)   
    # Calculate p-value
    dof = len(ebins)-1
    prob = 1 - stats.chi2.cdf(chi2sum_red, df=dof)
    print('chi2sum', chi2sum)
    print('chi2sum_red', chi2sum_red)
    print('p-values', prob)
    #return prob
    return chi2sum_red

# test-weights test for two weighted distributions
def test_weights(data1, data2, wei1, wei2, npix_on, npix_off):
    # Create the energy histogram
    ebins = np.logspace(2.5,5.0,20)
    ehist_on, edges = np.histogram(data1,bins=ebins,weights=wei1)
    var_on, edges = np.histogram(data1,bins=ebins,weights=np.power(wei1,2))
    sigma_on = np.sqrt(var_on)
    ehist_off,edges = np.histogram(data2,bins=ebins,weights=wei2)
    var_off, edges = np.histogram(data2,bins=ebins,weights=np.power(wei2,2))
    sigma_off = np.sqrt(var_off)
    # Check for zero or very small denominators
    denominator_on = np.where(sigma_on / np.sum(ehist_on) == 0, np.inf, sigma_on / np.sum(ehist_on))
    denominator_off = np.where(sigma_off / np.sum(ehist_off) == 0, np.inf, sigma_off / np.sum(ehist_off))
    # Calculate chi-squared sum with handled denominators
    chi2sum = 0
    chi2sum = np.sum(np.power((ehist_on/np.sum(ehist_on) - ehist_off/np.sum(ehist_off)), 2) / (np.power(denominator_on, 2) + np.power(denominator_off, 2)))
    chi2sum_red = chi2sum/(len(ebins)-1)

    if 0.0 <= chi2sum_red <= 10.0:
        plt.errorbar(edges[1:],ehist_on/np.sum(ehist_on),yerr=sigma_on/np.sum(ehist_on),label='$N_{on}$')
        plt.errorbar(edges[1:],ehist_off/np.sum(ehist_off),yerr=sigma_off/np.sum(ehist_off),label='$N_{off}$')
        plt.loglog()
        plt.legend()
        plt.xlabel("Energy bins")
        plt.ylabel("ehist/np.sum(ehist)")
        fig = pylab.figure(1)
        fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/test_weights_phy_ind_minus1_final_pix_20bins/test_weights_phy_ind_minus1_final_pix_20bins_'+str(round(chi2sum_red,1))+'.png', dpi=250)
        plt.close()

    print('chi2sum', chi2sum)
    print('chi2sum_red', chi2sum_red)
    return chi2sum_red

def test_weights_v2(data1, data2, wei1, wei2, fig_path, name):
    """
    Calculate the chi-squared statistic for weighted histograms of two datasets,
    normalize the weights, create histograms, and generate a plot if the reduced 
    chi-squared value is within a specified range.
    
    Parameters:
    - data1 (array-like): The first dataset.
    - data2 (array-like): The second dataset.
    - wei1 (array-like): Weights for the first dataset.
    - wei2 (array-like): Weights for the second dataset.
    - fig_path (str): Path to save the figure.
    - name (str): Name to be used in the figure file.
    
    Returns:
    - chi2sum_red (float): The reduced chi-squared value.
    """
    # Create the energy histogram bins
    ebins = np.logspace(1.5,5.5,21)
    
    # Normalize weights
    norm1 = np.sum(wei1)
    norm2 = np.sum(wei2)
    print("norm1/norm2", norm1, norm2)

    wei1_norm = wei1 / norm1
    wei2_norm = wei2 / norm2
    print("wei1_norm/wei2_norm", wei1_norm, wei2_norm)

    # Create histograms with normalized weights
    Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1_norm)
    S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1_norm, 2))
    Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2_norm)
    S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2_norm, 2))

    # Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1, density=True)
    # S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1, 2), density=True)
    # Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2, density=True)
    # S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2, 2), density=True)

    # Avoid divisions by zero
    valid_Wi_on = Wi_on > 0
    valid_Wi_off = Wi_off > 0

    # Calculate the denominators and handle zero denominators
    di2 = np.zeros_like(Wi_off, dtype=np.float64)
    di2[valid_Wi_on & valid_Wi_off] = Wi_off[valid_Wi_on & valid_Wi_off] * (S2i_on[valid_Wi_on & valid_Wi_off] / Wi_on[valid_Wi_on & valid_Wi_off] + S2i_off[valid_Wi_on & valid_Wi_off] / Wi_off[valid_Wi_on & valid_Wi_off])
    di2[~(valid_Wi_on & valid_Wi_off)] = np.inf

    # Calculate chi-squared sum
    chi2sum = np.sum(np.power((Wi_on - Wi_off), 2) / di2)
    chi2sum_red = chi2sum / (len(ebins) - 1)

    # if 0.0 <= chi2sum_red <= 10.0:
    #     # Plot the histograms with error bars
    #     plt.errorbar(edges[1:], Wi_on / np.sum(Wi_on), yerr=np.sqrt(S2i_on) / np.sum(Wi_on), fmt='o', label='$N_{on}$')
    #     plt.errorbar(edges[1:], Wi_off / np.sum(Wi_off), yerr=np.sqrt(S2i_off) / np.sum(Wi_off), fmt='o', label='$N_{off}$')
    #     plt.loglog()
    #     plt.legend()
    #     plt.xlabel("Energy bins")
    #     plt.ylabel("Wi/np.sum(Wi)")

    #     # Save the figure
    #     fig = pylab.figure(1)
    #     fig.savefig(f"{fig_path}{name}_{round(chi2sum_red, 1)}.png", dpi=250)
    #     plt.close()

    if 0.0 <= chi2sum_red <= 10.0:
        # Plot the histograms with error bars
        plt.errorbar(edges[1:], Wi_on, yerr=np.sqrt(S2i_on), fmt='o', label='$N_{on}$')
        plt.errorbar(edges[1:], Wi_off, yerr=np.sqrt(S2i_off), fmt='o', label='$N_{off}$')
        plt.loglog()
        plt.legend()
        plt.xlabel("Energy bins")
        plt.ylabel("Wi")

        # Save the figure
        fig = pylab.figure(1)
        fig.savefig(f"{fig_path}{name}_{round(chi2sum_red, 1)}.png", dpi=250)
        plt.close()

    #print('chi2sum:', chi2sum)
    print('chi2sum_red:', chi2sum_red)
    return chi2sum_red

# Get list of events and weights for a given pixel
def get_pixel_distribution(pixel):
    energies = [particle[0] for particle in pixel]
    weights = [particle[1] for particle in pixel]
    return np.array([energies, weights])

# Get an average distribution across the whole sky
def get_sky_distribution(pixel_list):
    energies = []
    weights = []
    for pixel in pixel_list:
        distribution = get_pixel_distribution(pixel)
        energies += distribution[0].tolist()
        weights += distribution[1].tolist()
    return np.array([energies, weights])

# Get the ring distribution of particles within a ring around a central pixel (Modified)
def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    return get_sky_distribution(pixel_list[particle_ring]), len(particle_ring)

# Get the pixel distribution of particles within a declination strip of the chosen pixel (Modified)
def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    theta, phi = hp.pix2ang(nside, pixel_number)
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    strip = hp.query_strip(nside, theta - num_pixels * d_theta, theta + num_pixels * d_theta)
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    strip = np.setdiff1d(strip, particle_ring)
    return get_sky_distribution(pixel_list[strip]), len(strip)

# Impose a fixed energy range on a distribution
def impose_energy_range(distribution, min_energy, max_energy):
    energies = distribution[0]
    weights = distribution[1]
    indices = np.where(np.logical_and(energies >= min_energy, energies <= max_energy))
    return np.array([energies[indices], weights[indices]])

# Perform Kolmorogov-Smirnov for a given set of particles, energy limits and strip width
def perform_kolmogorov_smirnov(particles, limits, width):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
    p_values = np.zeros(npix)
    for i in tqdm(range(npix)):
        strip_distribution = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = ks_weighted(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1])
        p_values[i] = results[1] * np.sign(results[0])
    return p_values

# Perform Chi-squared for a given set of particles, energy limits and strip width
def perform_chi_squared(particles, limits, width):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
    #p_values = np.zeros(npix)
    chi2sum = np.zeros(npix)
    for i in tqdm(range(npix)):
        strip_distribution, npix_off = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution, npix_on = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = chi_squared_weighted(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1], npix_on, npix_off)
        #p_values[i] = results
        chi2sum[i] = results
    return chi2sum

# # Histogram on one pixel
# def perform_histogram_on_pixel(pix_ind, particles, limits, width):
#     # Physical constants for scaling momentum
#     c = 299792458
#     e = 1.60217663 * 10 ** (-19)
#     m_p = 1.67262192 * 10 ** (-27)
#     npix = len(particles)
#     nside = hp.npix2nside(npix)
#     lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
#     upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
#     pixel_distribution, npix_on = get_ring_distribution(pix_ind, particles, nside, width)
#     pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
#     ebins = np.logspace(2.0,5.0,20)
#     plt.hist(pixel_distribution[0], bins=ebins, edgecolor='black', label='Energy histogram centered on pixel 1000')
#     plt.loglog()
#     plt.legend()
#     plt.xlabel('Energy bins')
#     plt.title('Histogram')
#     fig = pylab.figure(1)
#     fig.savefig('/home/aamarinp/Documents/ptracing-CosmicRay-analysis/figs/hist_on_pixel/hist_on_pixel_1000.png', dpi=250)
#     plt.close()

#     return None

# Perform test-weights for a given set of particles, energy limits and strip width
def perform_test_weights(particles, limits, width):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
    chi2sum = np.zeros(npix)
    for i in tqdm(range(npix)):
        strip_distribution, npix_off = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution, npix_on = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = test_weights(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1], npix_on, npix_off)
        chi2sum[i] = results
    return chi2sum

# Perform test-weights for a given set of particles, energy limits and strip width
def perform_test_weights_v2(particles, limits, width, fig_path, name):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(particles)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))
    chi2sum = np.zeros(npix)
    for i in tqdm(range(npix)):
        strip_distribution, _ = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution, _ = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = test_weights_v2(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1], fig_path, name)
        chi2sum[i] = results
    return chi2sum