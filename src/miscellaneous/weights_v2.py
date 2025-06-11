def create_weights_v2(nside, bins, obs_parameters, imposed_parameters,
                      physical_index, particle_dir, particle_file):
    particles_data = np.load(particle_dir + particle_file, allow_pickle=True)
    particles = particles_data['particles']
    npix = hp.nside2npix(nside)

    # Extracting energies from particles
    energies = particles[:, 2]  
    p_min, p_max = np.min(energies), np.max(energies)

     # Create bins
    bin_sizes = np.logspace(np.log10(p_min * 0.99), np.log10(p_max * 1.001), bins + 1)

    # Create a sky map for each bin, for weighing by energy
    final_maps = np.zeros((bins, npix))
    energy_bin_counts = np.zeros((npix, bins))
    pixel_counts = np.zeros(npix)

    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = np.digitize(p, bin_sizes)
        pixel_counts[final_pixel] += 1.0
        energy_bin_counts[final_pixel][p_bin-1] += 1.0

    for ipix in range(npix):
        if np.any(energy_bin_counts[ipix] == 0):
            continue
        pixnorm = 1.0 / pixel_counts[ipix]
        eweight = 1.0 / energy_bin_counts[ipix]  # Calculate eweight for each pixel
        eweight_norm = np.sum(eweight)
        if eweight_norm == 0 or np.isnan(eweight_norm):
            continue
        eweight /= eweight_norm
        for ebin in range(bins):
            final_maps[ebin][ipix] = pixnorm * eweight[ebin]

    reweighed_particles = [[] for i in range(npix)]

    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        p = item[2]
        bx, by, bz = item[3], item[4], item[5]
        p_bin = np.digitize(p, bin_sizes)
        uniform, dipole = imposed_parameters[0], imposed_parameters[1]
        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, bx, by, bz)
        direction_weight = final_maps[p_bin-1][final_pixel]
        momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1], physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)
        reweighed_particles[initial_pixel].append([p, momentum_weight * imposed_weight * obs_weight * direction_weight])

    max_length = max(len(sublist) for sublist in reweighed_particles)
    reweighed_particles_equalized = [sublist + [[p_min, 0.0]] * (max_length - len(sublist)) for sublist in reweighed_particles]
    
    return reweighed_particles_equalized