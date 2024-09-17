
def create_weights(nside, bins, obs_parameters, imposed_parameters,
                   physical_index, particle_dir, particle_file):

    # Find corresponding particle data and load it
    particles_data = np.load(particle_dir + particle_file, allow_pickle=True)
    particles = particles_data['particles']

    npix = hp.nside2npix(nside)

    # Create energy binning scheme
    p_max, p_min = 0, sys.maxsize

    # Determine max and min energy
    for item in particles:
        if item[2] < p_min:
            p_min = item[2]
        elif item[2] > p_max:
            p_max = item[2]

    # Create bins
    bin_sizes = np.logspace(np.log10(p_min * 0.99), np.log10(p_max * 1.001),
                            bins + 1)

    # Create a sky map for each bin, for weighing by energy
    final_maps = np.zeros((bins, npix))
    energy_bin_counts = np.zeros((npix,bins))
    pixel_counts = np.zeros(npix)

    reweighed_particles = [[] for i in range(npix)]

    # Populate initial and final maps
    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = numpy.digitize(p, bin_sizes)

        pixel_counts[final_pixel] += 1.0
        energy_bin_counts[final_pixel][p_bin] += 1.0

    for ipix in range(npix):
        pixnorm = 1.0/pixel_counts[ipix]
        eweight = 1.0/energy_bin_counts

        eweight_norm = np.sum(eweight)
        eweight *= 1.0/eweight_norm

        for ebin in range(bins):
            final_maps[p_bin][ipix] += pixnorm*eweight[p_bin]

    # Go back through the data and reweigh the initial map. Save the individual
    # particle data for statistical testing
    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        p = item[2]
        b = item[3]
        p_bin = numpy.digitize(p, bin_sizes)

        uniform = imposed_parameters[0]
        dipole = imposed_parameters[1]

        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, b)
        direction_weight = final_maps[p_bin][final_pixel]
        momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1],
                                          physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)

        reweighed_particles[initial_pixel].append([p, momentum_weight *
                                                  imposed_weight * obs_weight * direction_weight])
    return reweighed_particles


