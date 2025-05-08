"""
This script processes cosmic ray particle data to generate weight maps based on energy distributions, directional factors, and observational parameters. 

Inputs:
- HEALPix resolution parameter (nside)
- Number of energy bins (bins)
- Observational parameters for weight calculation
- Imposed weight parameters (uniform and dipole factors)
- Physical power-law index
- Particle data file containing energy, direction, and pixel assignments
- Output file to save the processed weights

Outputs:
- A compressed file containing the reweighted particle data
- Various weight components including momentum, direction, observational, and imposed weights

The code first reads the particle data, bins the energies, and computes normalized weights per pixel. It then applies different weighting factors based on physics-driven models and saves the results for further analysis.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Function to calculate the cosine of the angle between a dipole vector and pixel positions in a HEALPix map
def cos_dipole_f(nside, pix, bx, by, bz):
    pxf, pyf, pzf = hp.pix2vec(nside, pix)  # Convert pixel index to 3D coordinates
    return -(pxf * bx + pyf * by + pzf * bz) / \
        (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1.e-16) / \
        np.sqrt(bx * bx + by * by + bz * bz)  # Normalize by dipole magnitude

# Function to compute the probability density function (PDF) of a power-law distribution
def powerlaw_pdf(x, x_min, x_max, power):
    x_min_g, x_max_g = x_min ** (power + 1.), x_max ** (power + 1.)
    if power == -1.0:
        return x ** power / np.log(x_max / x_min)  # Special case for power = -1
    else:
        return (power + 1.) / (x_max_g - x_min_g) * x ** power  # General case

# Function to compute the weight of a given value based on a power-law PDF
def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)

# Function to compute observational weights for particles based on a Gaussian distribution
def observational_weight(particle_energy, obs_parameters):
    if obs_parameters[0] == -1 and obs_parameters[1] == -1:
        return 1  # Default weight if parameters are not defined
    else:
        c = 299792458  # Speed of light in m/s
        e = 1.60217663 * 10 ** (-19)  # Elementary charge in Coulombs
        m_p = 1.67262192 * 10 ** (-27)  # Proton mass in kg
        #energy_factor = 1 / (m_p * c * c / (e * 10 ** 12))
        energy_factor = 1.

        sigma = obs_parameters[0]  # Standard deviation of Gaussian
        mid_energy = np.log10(obs_parameters[1] * energy_factor)  # Mean value of Gaussian
        logged_energy = np.log10(particle_energy)

        return np.exp(-0.5 * np.square((logged_energy - mid_energy) / sigma)) / (sigma * np.sqrt(2 * np.pi))

# Function to generate normalized weights for cosmic ray particles
def compute_particle_weights(nside, bins, imposed_parameters, physical_index, particle_dir, particle_file, output_file):
    try:
        particles_data = np.load(particle_dir + particle_file, allow_pickle=True)  # Load particle data
    except FileNotFoundError:
        print("Error: File not found.")
        return 1  
    except Exception as e:
        print(f"Error: {e}")
        return 2 

    particles = particles_data['particles']  # Extract particle data
    npix = hp.nside2npix(nside)  # Compute total number of pixels

    energies = particles[:, 2]  # Extract particle energies
    energy_bins = np.logspace(np.log10(min(energies)), np.log10(max(energies)), bins + 1)  # Create logarithmic energy bins
    print('max energy', max(energies)) # ~500000
    print('min energy', min(energies)) # ~32.0
    print('np.log10(min(energies))', np.log10(min(energies))) # ~1.505
    print('np.log10(max(energies))', np.log10(max(energies))) # ~5.699

    final_maps = np.zeros((bins, npix))  # Initialize weight maps
    energy_bin_counts = np.zeros((npix, bins))  # Store energy bin counts per pixel
    pixel_counts = np.zeros(npix)  # Count particles per pixel

    # Loop through all particles to count pixels and energy bins
    for item in particles:
        final_pixel = int(item[1])  # Pixel index where the particle is mapped
        p = item[2]  # Energy of the particle
        p_bin = np.digitize(p, energy_bins) - 1  # Determine the appropriate energy bin
        pixel_counts[final_pixel] += 1.0  # Update particle count for the pixel
        if 0 <= p_bin < bins:
            energy_bin_counts[final_pixel, p_bin] += 1.0  # Increment count for the energy bin

    # Normalize weights by pixel and energy bin
    for ipix in range(npix):
        # Calculate weights for each energy bin in the pixel
        eweight = 1.0 / energy_bin_counts[ipix]
        # Replace infinite values with zero to avoid computational errors
        eweight[np.isinf(eweight)] = 0.0
        # Normalize energy weights by their sum
        eweight_norm = np.sum(eweight)
        # Calculate normalization factor for the pixel based on the particle count
        if eweight_norm > 0:
            eweight /= eweight_norm  
        # Store the final normalized weight in the final_maps array
        for ebin in range(bins):
            final_maps[ebin, ipix] = eweight[ebin]

    reweighed_particles = [[] for _ in range(npix)]  # Initialize reweighted particles list

    # Calculate weights for each particle based on direction, energy, and imposed factors
    for item in particles:
        initial_pixel = int(item[0])  # The original pixel where the particle originated
        final_pixel = int(item[1])  # The pixel to which the particle is mapped
        # final_pixel = np.random.randint(npix)
        p = item[2]  # The energy of the particle
        # The directional components of B-field
        # The assumption is that B at final radius is constant
        bx, by, bz = item[3], item[4], item[5]  
        p_bin = np.digitize(p, energy_bins) - 1  # Determine which energy bin the particle falls into

        imposed_weight = 1.0 + imposed_parameters[1] * cos_dipole_f(nside, final_pixel, bx, by, bz)
        direction_weight = final_maps[p_bin, final_pixel] if 0 <= p_bin < bins else 0
        momentum_weight = weight_powerlaw(p, energy_bins[0], energy_bins[-1], physical_index, -1)

        total_weight = momentum_weight * imposed_weight * direction_weight
        reweighed_particles[initial_pixel].append([p, total_weight])  # Store final weight

    return convert_to_numpy_array(reweighed_particles)  

def convert_to_numpy_array(reweighed_particles):
    max_length = max(len(sublist) for sublist in reweighed_particles)
    padded_array = np.array(
        [sublist + [[0, 0]] * (max_length - len(sublist)) for sublist in reweighed_particles],
        dtype=np.float32
    )
    return padded_array

def save_results(reweighed_particles_array, output_file):
    np.savez_compressed(output_file, reweighed_particles=reweighed_particles_array)
    print(f"Results saved to {output_file}")

particle_dir = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/"
particle_file = "nside=32.npz"
output_file = particle_dir+"real_mapping_phyind-2p6_all-weights_nside=32.npz"

# Execute function and save results
result = compute_particle_weights(32, 120, [1.0, 0.001], -2.6, particle_dir, particle_file, output_file)
save_results(result, output_file)