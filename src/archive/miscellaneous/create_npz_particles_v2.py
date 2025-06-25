import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def cos_dipole_f(nside, pix, bx, by, bz):
    pxf, pyf, pzf = hp.pix2vec(nside, pix)
    return -(pxf * bx + pyf * by + pzf * bz) / \
        (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1.e-16) / \
        np.sqrt(bx * bx + by * by + bz * bz)

def powerlaw_pdf(x, x_min, x_max, power):
    x_min_g, x_max_g = x_min ** (power + 1.), x_max ** (power + 1.)
    if power == -1.0:
        return x ** power / np.log(x_max / x_min)
    else:
        return (power + 1.) / (x_max_g - x_min_g) * x ** power

def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)

def observational_weight(particle_energy, obs_parameters):
    if obs_parameters[0] == -1 and obs_parameters[1] == -1:
        return 1
    else:
        c = 299792458
        e = 1.60217663 * 10 ** (-19)
        m_p = 1.67262192 * 10 ** (-27)
        energy_factor = 1 / (m_p * c * c / (e * 10 ** 12))

        sigma = obs_parameters[0]
        mid_energy = np.log10(obs_parameters[1] * energy_factor)
        logged_energy = np.log10(particle_energy)

        return np.exp(-0.5 * np.square((logged_energy - mid_energy) / sigma)) \
            / (sigma * np.sqrt(2 * np.pi))

def create_weights_v3_no_padding(nside, bins, obs_parameters, imposed_parameters, physical_index, particle_dir, particle_file, output_file):
    try:
        particles_data = np.load(particle_dir + particle_file, allow_pickle=True)
    except FileNotFoundError:
        print("Error: File not found.")
        return 1  
    except Exception as e:
        print(f"Error: {e}")
        return 2 
    
    particles = particles_data['particles']
    npix = hp.nside2npix(nside)

    energies = particles[:, 2]
    p_min, p_max = np.min(energies), np.max(energies)
    energy_bins = np.logspace(1.5, 5.5, bins + 1)

    final_maps = np.zeros((bins, npix))
    energy_bin_counts = np.zeros((npix, bins))
    pixel_counts = np.zeros(npix)

    # Count particles by pixel and energy bin
    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = np.digitize(p, energy_bins) - 1
        pixel_counts[final_pixel] += 1.0
        if 0 <= p_bin < bins:
            energy_bin_counts[final_pixel, p_bin] += 1.0

    # Normalize weights by pixel and energy
    for ipix in range(npix):
        pixnorm = 1.0 / pixel_counts[ipix] if pixel_counts[ipix] > 0 else 0
        eweight = 1.0 / energy_bin_counts[ipix]
        eweight[np.isinf(eweight)] = 0.0
        eweight_norm = np.sum(eweight)
        if eweight_norm > 0:
            eweight /= eweight_norm      
        for ebin in range(bins):
            final_maps[ebin, ipix] = pixnorm * eweight[ebin]

    reweighed_particles = [[] for _ in range(npix)]
    total_w = []
    momentum_weights = []
    direction_weights = []
    obs_weights = []
    imposed_weights = []

    # Calculate weights for each particle
    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        #final_pixel = np.random.randint(npix)
        p = item[2]
        bx, by, bz = item[3], item[4], item[5]
        p_bin = np.digitize(p, energy_bins) - 1

        uniform, dipole = imposed_parameters
        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, bx, by, bz)
        direction_weight = final_maps[p_bin, final_pixel] if 0 <= p_bin < bins else 0
        momentum_weight = weight_powerlaw(p, energy_bins[0], energy_bins[-1], physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)

        # Normalize each component independently
        # imposed_weight /= np.sum(imposed_weight) if np.sum(imposed_weight) > 0 else 1
        # direction_weight /= np.sum(direction_weight) if np.sum(direction_weight) > 0 else 1
        # momentum_weight /= np.sum(momentum_weight) if np.sum(momentum_weight) > 0 else 1
        # obs_weight /= np.sum(obs_weight) if np.sum(obs_weight) > 0 else 1

        # Store individual weights for histogram analysis
        momentum_weights.append(momentum_weight)
        direction_weights.append(direction_weight)
        obs_weights.append(obs_weight)
        imposed_weights.append(imposed_weight)

        # Calculate the total weight using normalized components
        total_weight = momentum_weight * imposed_weight * obs_weight * direction_weight
        reweighed_particles[initial_pixel].append([p, total_weight])
        total_w.append(total_weight)

    # Convert weights to arrays
    momentum_weights = np.array(momentum_weights)
    direction_weights = np.array(direction_weights)
    obs_weights = np.array(obs_weights)
    imposed_weights = np.array(imposed_weights)
    total_w = np.array(total_w)

    # Plot all histograms in a single figure
    # plot_all_histograms(momentum_weights, direction_weights, obs_weights, imposed_weights, total_w)

    # Convert to NumPy array and save results
    reweighed_particles_array = convert_to_numpy_array(reweighed_particles)
    save_results(reweighed_particles_array, output_file)

    return reweighed_particles_array

def plot_all_histograms(momentum_weights, direction_weights, obs_weights, imposed_weights, total_w):
    """Plot all weight histograms in a single figure with subplots."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    weights_data = [
        (momentum_weights, "Momentum Weight"),
        (direction_weights, "Direction Weight"),
        (obs_weights, "Observational Weight"),
        (imposed_weights, "Imposed Weight"),
        (total_w, "Total Weight")
    ]

    for ax, (weights, title) in zip(axes.flatten(), weights_data):
        ax.hist(weights, bins=50, log=True)
        ax.set_title(f"{title} Histogram")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    fig.delaxes(axes[2, 1])
    plt.tight_layout()
    plt.show()

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
particle_file = "nside=16.npz"
output_file = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/rewei_part_nside=16_bins=120_pwrind=-1_real-mapping.npz"

# Execute function and save results
result = create_weights_v3_no_padding(16, 120, [-1, -1], [1.0, 0.003], -1, particle_dir, particle_file, output_file)



