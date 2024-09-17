#!/usr/local/bin/python
import numpy as np
import healpy as hp

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
    # If observational weighting is not being used, return uniform
    if obs_parameters[0] == -1 and obs_parameters[1] == -1:
        return 1
    else:
        # Physical constants for scaling energy
        c = 299792458
        e = 1.60217663 * 10 ** (-19)
        m_p = 1.67262192 * 10 ** (-27)
        energy_factor = 1 / (m_p * c * c / (e * 10 ** 12))

        # Parameters for changing the shape of the distribution
        sigma = obs_parameters[0]
        mid_energy = np.log10(obs_parameters[1] * energy_factor)

        logged_energy = np.log10(particle_energy)

        return np.exp(-0.5 * np.square((logged_energy - mid_energy) / sigma)) \
            / (sigma * np.sqrt(2 * np.pi))

def create_weights_v3(nside, bins, obs_parameters, imposed_parameters, physical_index, particle_dir, particle_file):
    particles_data = np.load(particle_dir + particle_file, allow_pickle=True)
    particles = particles_data['particles']
    npix = hp.nside2npix(nside)

    # Extracting energies from particles
    energies = particles[:, 2]
    p_min, p_max = np.min(energies), np.max(energies)

    # Create bins
    bin_sizes = np.logspace(1.5, 5.5, bins + 1)

    # Initialize maps and counts
    final_maps = np.zeros((bins, npix))
    energy_bin_counts = np.zeros((npix, bins))
    pixel_counts = np.zeros(npix)

    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = np.digitize(p, bin_sizes) - 1
        pixel_counts[final_pixel] += 1.0
        if 0 <= p_bin < bins:
            energy_bin_counts[final_pixel, p_bin] += 1.0

    for ipix in range(npix):
        pixnorm = 1.0 / pixel_counts[ipix]
        eweight = 1.0 / energy_bin_counts[ipix]
        eweight[np.isinf(eweight)] = 0.0
        eweight_norm = np.sum(eweight)
        if eweight_norm > 0:
            eweight /= eweight_norm      
        for ebin in range(bins):
            final_maps[ebin, ipix] = pixnorm * eweight[ebin]

    reweighed_particles = [[] for _ in range(npix)]

    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        #final_pixel = np.random.randint(npix)
        p = item[2]
        bx, by, bz = item[3], item[4], item[5]
        p_bin = np.digitize(p, bin_sizes) - 1
        uniform, dipole = imposed_parameters
        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, bx, by, bz)
        direction_weight = final_maps[p_bin, final_pixel] if 0 <= p_bin < bins else 0
        momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1], physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)
        total_weight = momentum_weight * imposed_weight * obs_weight * direction_weight
        reweighed_particles[initial_pixel].append([p, total_weight])

    max_length = max(len(sublist) for sublist in reweighed_particles)
    reweighed_particles_equalized = [sublist + [[p_min, 0.0]] * (max_length - len(sublist)) for sublist in reweighed_particles]
    return reweighed_particles_equalized

particle_dir = "/home/aamarinp/Documents/ptracing-CosmicRay-analysis/data/particles/"
particle_file = "nside=8.npz"
weighted_particles = create_weights_v3(8, 20, [-1,-1], [1.0,0.003], -2.6, particle_dir, particle_file)
weighted_particles = np.array(weighted_particles)
np.savez_compressed(particle_dir + "phyindex_2p6_nside8_realData" + ".npz", data=weighted_particles)