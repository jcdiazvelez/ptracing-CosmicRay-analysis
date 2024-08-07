import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import random
import scipy.stats 

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

def generate_synthetic_uniform_data(low, high, sample_size):
    data = np.random.uniform(low, high, sample_size)
    return data

def generate_log_uniform_data(low, high, sample_size):
    if low <= 0:
        raise ValueError("low limit must be higher than 0")
    
    log_low = np.log10(low)
    log_high = np.log10(high)
    #print('low/high' ,log_low, log_high)
    
    log_data = np.random.uniform(log_low, log_high, sample_size)
    data = np.power(10, log_data)
    
    return log_data

def generate_synthetic_normal_data(mean, std_dev, sample_size):
    data = np.random.normal(mean, std_dev, sample_size)
    #return (data - np.min(data)) / (np.max(data) - np.min(data)) if np.max(data) > np.min(data) else data
    return data

def generate_normal_weights(mean, std_dev, sample_size):
    weights = np.random.normal(mean, std_dev, sample_size)
    return (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) if np.max(weights) > np.min(weights) else weights
    #return weights

def generate_log_normal_weights(mean, std_dev, sample_size):
    if mean <= 0:
        raise ValueError("The mean must be greater than 0 for the logarithmic scale.")
    
    # Generate data on a logarithmic scale
    log_mean = np.log10(mean)
    log_std_dev = np.log10(std_dev) if std_dev > 0 else 0
    log_data = np.random.normal(log_mean, log_std_dev, sample_size)
    
    # Transform from logarithmic to linear scale
    linear_data = np.power(10, log_data)
    
    # Ensure weights are positive and normalize them
    weights = (linear_data - np.min(linear_data)) / (np.max(linear_data) - np.min(linear_data))
    return weights
    #return np.ones(sample_size)

def generate_synthetic_poisson_data(lambda_val, sample_size):
    data = np.random.poisson(lambda_val, sample_size)
    #return data / np.max(data) if np.max(data) > 0 else data
    return data

def generate_poisson_weights(lambda_val, sample_size):
    weights = np.random.poisson(lambda_val, sample_size)
    return weights / np.max(weights)
    #return weights

def generate_log_poisson_weights(lambda_val, sample_size):
    if lambda_val <= 0:
        raise ValueError("The lambda value must be greater than 0.")
    
    # Generate data on a logarithmic scale
    log_lambda = np.log10(lambda_val)
    log_data = np.random.uniform(0, log_lambda, sample_size)
    
    # Transform from logarithmic to linear scale
    linear_data = np.power(10, log_data)
    
    # Generate Poisson weights
    weights = np.random.poisson(linear_data)
    
    # Normalization to ensure weights are between 0 and 1
    weights = weights / np.max(weights)
    #return weights
    return np.ones(sample_size)

def test_weights_v2(data1, data2, wei1, wei2):
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    #print('min_val, max_val', min_val, max_val)
    bins_count = 21

    if min_val <= 0:
        min_val = 1e-10  # Small positive value

    ebins = np.logspace(np.log10(min_val), np.log10(max_val), bins_count)
    #print('np.log10(min_val), np.log10(max_val)', np.log10(min_val), np.log10(max_val))

    # Normalize data
    # norm_data1 = np.sum(data1)
    # norm_data2 = np.sum(data2)
    # data1_norm = data1 / norm_data1
    # data2_norm = data2 / norm_data2
    # print("data1_norm min/max", np.min(data1_norm), np.max(data1_norm))
    # print("data2_norm min/max", np.min(data2_norm), np.max(data2_norm))

    # Normalize weights
    norm1 = np.sum(wei1)
    norm2 = np.sum(wei2)
    wei1_norm = wei1 / norm1
    wei2_norm = wei2 / norm2
    print("wei1_norm min/max", np.min(wei1_norm), np.max(wei1_norm))
    print("wei2_norm min/max", np.min(wei2_norm), np.max(wei2_norm))

    Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1_norm)
    S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1_norm, 2))
    Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2_norm)
    S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2_norm, 2))
    print('Wi_on/Wi_off', Wi_on, Wi_off)

    # Wi_on, edges = np.histogram(data1, bins=ebins, weights=wei1)
    # S2i_on, _ = np.histogram(data1, bins=ebins, weights=np.power(wei1, 2))
    # Wi_off, _ = np.histogram(data2, bins=ebins, weights=wei2)
    # S2i_off, _ = np.histogram(data2, bins=ebins, weights=np.power(wei2, 2))

    valid_Wi_on = Wi_on > 0
    valid_Wi_off = Wi_off > 0

    di2 = np.zeros_like(Wi_off, dtype=np.float64)
    di2[valid_Wi_on & valid_Wi_off] = Wi_off[valid_Wi_on & valid_Wi_off] * (
        S2i_on[valid_Wi_on & valid_Wi_off] / Wi_on[valid_Wi_on & valid_Wi_off] + 
        S2i_off[valid_Wi_on & valid_Wi_off] / Wi_off[valid_Wi_on & valid_Wi_off])
    di2[~(valid_Wi_on & valid_Wi_off)] = np.inf

    # chi2sum = np.power((Wi_on - Wi_off), 2) / di2 and then return chi2sum not the reduced one
    chi2sum = np.sum(np.power((Wi_on - Wi_off), 2) / di2)
    chi2sum_red = chi2sum / (len(ebins) - 1)
    print('Wi_on - Wi_off', Wi_on - Wi_off)

    return chi2sum_red

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
    print('1 min/max', np.min(wei1_norm), np.max(wei1_norm))
    print('2 min/max', np.min(wei2_norm), np.max(wei2_norm))

    # Histogram
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
    # chi2sum = np.power((Wi_on - Wi_off), 2) / di2 and then return chi2sum not the reduced one
    chi2 = np.sum(np.power((Wi_on - Wi_off), 2) / di2)
    return chi2

def plot_histograms_with_pdfs(data1, weights1, data2, weights2, distribution, params1, params2, title):
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for the first dataset
    count1, bins1, ignored1 = plt.hist(data1, bins=21, density=False, alpha=0.5, color='g', weights=weights1, label='Dataset 1')
    
    # Plot histogram for the second dataset
    count2, bins2, ignored2 = plt.hist(data2, bins=21, density=False, alpha=0.5, color='b', weights=weights2, label='Dataset 2')
    
    # Calculate the width of each bin
    bin_width1 = bins1[1] - bins1[0]
    bin_width2 = bins2[1] - bins2[0]
    
    # # Plot PDF/PMF for the first distribution
    # if distribution == 'poisson':
    #     lambda_val = params1['lambda']
    #     plt.plot(bins1[:-1], poisson.pmf(bins1[:-1], lambda_val) * np.sum(weights1) * bin_width1, 'r-', lw=2, label=f'Poisson PMF ($\\lambda$={lambda_val}) - Dataset 1')
    # elif distribution == 'normal':
    #     mean1, std_dev1 = params1['mean'], params1['std_dev']
    #     plt.plot(bins1[:-1], norm.pdf(bins1[:-1], mean1, std_dev1) * np.sum(weights1) * bin_width1, 'r-', lw=2, label=f'Normal PDF ($\\mu$={mean1}, $\\sigma$={std_dev1}) - Dataset 1')

    # # Plot PDF/PMF for the second distribution
    # if distribution == 'poisson':
    #     lambda_val = params2['lambda']
    #     plt.plot(bins2[:-1], poisson.pmf(bins2[:-1], lambda_val) * np.sum(weights2) * bin_width2, 'm-', lw=2, label=f'Poisson PMF ($\\lambda$={lambda_val}) - Dataset 2')
    # elif distribution == 'normal':
    #     mean2, std_dev2 = params2['mean'], params2['std_dev']
    #     plt.plot(bins2[:-1], norm.pdf(bins2[:-1], mean2, std_dev2) * np.sum(weights2) * bin_width2, 'm-', lw=2, label=f'Normal PDF ($\\mu$={mean2}, $\\sigma$={std_dev2}) - Dataset 2')
    
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_skymap(chi2_values, nside):
    npix = hp.nside2npix(nside)
    skymap = np.full(npix, hp.UNSEEN)
    for idx, chi2 in chi2_values:
        skymap[idx] = chi2
    return skymap

###################################################################################
# Global parameters
sample_size = 5000

# Parameters for Uniform data
low1 = 5.287760354365922 
high1 = 14.985871050425096
low2= 5.3
high2 = 15.0

# Parameters for Poisson
lambda1 = 9.8
lambda2 = 10.1

# Parameters for Normal
mean1 = 9.4
std_dev1 = 1.11
mean2 = 10.2
std_dev2 = 1.05

mean3 = 1.0
std_dev3 = 0.5
mean4 = 1.0
std_dev4 = 0.5
###################################################################################

###################################################################################
# Generate synthetic Uniform data
# data1_uniform = generate_log_uniform_data(low1, high1, sample_size)
# data2_uniform = generate_log_uniform_data(low2, high2, sample_size)
#print("data1_uniform min/max", np.min(data1_uniform), np.max(data1_uniform))
#print("data2_uniform min/max", np.min(data2_uniform), np.max(data2_uniform))

# Generate synthetic Poisson data
# data1_poisson = generate_synthetic_poisson_data(lambda1, sample_size)
# data2_poisson = generate_synthetic_poisson_data(lambda2, sample_size)
#print("data1_poisson min/max", np.min(data1_poisson), np.max(data1_poisson))
#print("data2_poisson min/max", np.min(data2_poisson), np.max(data2_poisson))

# Generate synthetic Normal data
# data1_normal = generate_synthetic_normal_data(mean1, std_dev1, sample_size)
# data2_normal = generate_synthetic_normal_data(mean2, std_dev2, sample_size)
#print("data1_normal min/max", np.min(data1_normal), np.max(data1_normal))
#print("data2_normal min/max", np.min(data2_normal), np.max(data2_normal))
###################################################################################

###################################################################################
# Generate weights using Poisson distribution and normalize to [0, 1]
# wei1_poisson = generate_log_poisson_weights(lambda1, sample_size)
# wei2_poisson = generate_log_poisson_weights(lambda2, sample_size)
#print("wei1_poisson min/max", np.min(wei1_poisson), np.max(wei1_poisson))
#print("wei2_poisson min/max", np.min(wei2_poisson), np.max(wei2_poisson))

# Generate weights using Normal distribution and normalize to [0, 1]
# wei1_normal = generate_normal_weights(mean3, std_dev3, sample_size)
# wei2_normal = generate_normal_weights(mean4, std_dev4, sample_size)
#print("wei1_normal min/max", np.min(wei1_normal), np.max(wei1_normal))
#print("wei2_normal min/max", np.min(wei2_normal), np.max(wei2_normal))
###################################################################################

###################################################################################
# Generate an array with random normal distributions at the boundary layer
ext_boundary_dist = []
for _ in range(sample_size):
    energies = generate_log_uniform_data(low1, high1, sample_size)
    # weight_powerlaw(x, x_min, x_max, g, power)
    # weights = generate_normal_weights(mean3, std_dev3, sample_size)
    weights = weight_powerlaw(energies, low1, high1, -2, -1)
    ext_boundary_dist.append([energies, weights])
#print('ext_boundary_dist', ext_boundary_dist)

# Randomly map the boundary distribution array to the inner boundary
int_boundary_dist = ext_boundary_dist.copy()
random.shuffle(int_boundary_dist)
#print('ext/int' , [sublist[1] for sublist in ext_boundary_dist], [sublist[1] for sublist in int_boundary_dist])

# Calculate chi2
chi2_concat = []
for i in range(sample_size):
    data1 = int_boundary_dist[i][0] 
    data2 = int_boundary_dist[(i + 2) % sample_size][0] 
    wei1 = int_boundary_dist[i][1]  
    wei2 = int_boundary_dist[(i + 3) % sample_size][1] 
    chi2 = test_weights_v3(data1, data2, wei1, wei2)
    chi2_concat.append(chi2)

#print('length', len(chi2_concat))    
#print('chi2', chi2)
#plot_histograms_with_pdfs(data1, wei1, data2, wei2, 'normal', {'mean': mean3, 'std_dev': std_dev3}, {'mean': mean3, 'std_dev': std_dev3}, 'Histogram')
xbins = np.logspace(-2, 3, 100)
hist, edges = np.histogram(chi2_concat, bins=xbins, density=True)
rv = scipy.stats.chi2(20)
plt.figure(figsize=(10, 6))
plt.plot(xbins, rv.pdf(xbins), 'k-', lw=2, label='pdf')
plt.plot(edges[1:], hist)
plt.yscale('log')
plt.xscale('log')
plt.title('Chi2 PDF')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

###################################################################################

###################################################################################
# # Calculate chi2sum_red for Uniform with Poisson weights
# chi2sum_red_uniform_poisson= test_weights_v2(data1_uniform, data2_uniform, wei1_poisson, wei2_poisson)
# print('Reduced Chi-squared (Uniform/Poisson):', chi2sum_red_uniform_poisson)
# # Plot Uniform with Normal weights histograms and PDFs
# plot_histograms_with_pdfs(data1_uniform, wei1_poisson, data2_uniform, wei2_poisson, 'poisson', {'lambda': lambda1}, {'lambda': lambda2}, 'Histogram and Poisson PMF')

# # Calculate chi2sum_red for Poisson
# chi2sum_red_poisson = test_weights_v2(data1_poisson, data2_poisson, wei1_poisson, wei2_poisson)
# print('Reduced Chi-squared (Poisson):', chi2sum_red_poisson)
# # Plot Poisson histograms and PDFs
# plot_histograms_with_pdfs(data1_poisson, wei1_poisson, data2_poisson, wei2_poisson, 'poisson', {'lambda': lambda1}, {'lambda': lambda2}, 'Histogram and Poisson PMF')

# Calculate chi2sum_red for Uniform with Normal weights
# chi2sum_red_uniform_normal = test_weights_v2(data1_uniform, data2_uniform, wei1_normal, wei2_normal)
# print('Reduced Chi-squared (Uniform/Normal):', chi2sum_red_uniform_normal)
# Plot Uniform with Normal weights histograms and PDFs
# plot_histograms_with_pdfs(data1_uniform, wei1_normal, data2_uniform, wei2_normal, 'normal', {'mean': mean1, 'std_dev': std_dev1}, {'mean': mean2, 'std_dev': std_dev2}, 'Histogram and Normal PDF')

# # Calculate chi2sum_red for Normal
# chi2sum_red_normal = test_weights_v2(data1_normal, data2_normal, wei1_normal, wei2_normal)
# print('Reduced Chi-squared (Normal):', chi2sum_red_normal)
# # Plot Normal histograms and PDFs
# plot_histograms_with_pdfs(data1_normal, wei1_normal, data2_normal, wei2_normal, 'normal', {'mean': mean1, 'std_dev': std_dev1}, {'mean': mean2, 'std_dev': std_dev2}, 'Histogram and Normal PDF')
###################################################################################

# # Create skymap for visualization (example)
# nside = 64  # This determines the resolution of the skymap
# npix = hp.nside2npix(nside)
# chi2_values = [(i, chi2sum_red_normal) for i in range(npix)]

# # Create the skymap
# skymap = create_skymap(chi2_values, nside)

# # Plot the skymap
# hp.mollview(skymap, title="Skymap of Reduced Chi-squared Values (Normal)")
# plt.show()
