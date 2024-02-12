import numpy as np
import healpy as hp
import scipy.stats as stat
from tqdm import tqdm
from scipy import stats


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
def chi_squared_weighted(data1, data2, wei1, wei2):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    # Create the energy histogram
    ebins = np.logspace(0,3,20)
    ehist_on,edges = np.histogram(data1,bins=ebins,weights=wei1)
    var_on, edges = np.histogram(data1,bins=ebins,weights=np.power(wei1,2))
    sigma_on = np.sqrt(var_on)
    ehist_off,edges = np.histogram(data2,bins=ebins,weights=wei2)
    var_off, edges = np.histogram(data2,bins=ebins,weights=np.power(wei2,2))
    sigma_off = np.sqrt(var_off)
    npix_on = 1
    npix_off = 50
    # Check for zero or very small denominators
    denominator_on = np.where(sigma_on / npix_on <= 0, np.inf, sigma_on / npix_on)
    denominator_off = np.where(sigma_off / npix_off <= 0, np.inf, sigma_off / npix_off)
    # Calculate chi-squared sum with handled denominators
    chi2sum = 0
    chi2sum = np.sum(np.power(ehist_on / npix_on - ehist_off / npix_off, 2) / (np.power(denominator_on, 2) + np.power(denominator_off, 2)))
    # Calculate p-value
    dof = len(ebins)-1
    prob = 1 - stats.chi2.cdf(chi2sum, df=dof)
    return prob

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


def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    return get_sky_distribution(pixel_list[particle_ring])


# Get the pixel distribution of particles within a declination strip of the chosen pixel
def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    theta, phi = hp.pix2ang(nside, pixel_number)
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    strip = hp.query_strip(nside, theta - num_pixels * d_theta, theta + num_pixels * d_theta)
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    strip = np.setdiff1d(strip, particle_ring)
    return get_sky_distribution(pixel_list[strip])


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

    p_values = np.zeros(npix)
    for i in tqdm(range(npix)):
        strip_distribution = get_strip_distribution(i, particles, nside, width)
        strip_distribution = impose_energy_range(strip_distribution, lower, upper)
        pixel_distribution = get_ring_distribution(i, particles, nside, width)
        pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
        results = chi_squared_weighted(pixel_distribution[0], strip_distribution[0],
                              pixel_distribution[1], strip_distribution[1])
        p_values[i] = results

    return p_values