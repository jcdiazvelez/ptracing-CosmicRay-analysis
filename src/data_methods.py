import glob
import sys
import numpy as np
import healpy as hp
from multiprocessing import Pool
from PathSegment import PathSegment


# Create particle data from raw traces
def create_particles(nside, particle_dir, particle_file, raw_dir):
    # Set target radius
    radius = 50000

    # Prepare filenames
    filename = "*.npz"
    path = raw_dir + "/" + filename

    # Prepare file names for processing
    files = sorted(glob.glob(path))
    n_files = len(files)

    # Use 16 worker processes
    pool = Pool(processes=16)

    # Create pool input for direction data map
    pool_input = []
    for i in range(n_files):
        pool_input.append((files[i], nside, radius))

    # Generate and flatten direction data
    direction_data = pool.starmap(process_particle_data, pool_input)
    direction_data = np.array([ent for sublist in
                               direction_data for ent in sublist])

    # Save produced data
    output_name = particle_dir + particle_file
    print("saving %s" % output_name)
    np.savez_compressed(output_name, particles=direction_data)


# Process data files and populate initial and final sky maps
def process_particle_data(filename, nside, radius):
    file = np.load(filename)

    data_array = []

    for key in file:
        try:
            # Get particle track
            particle = file[key]

            if len(particle) < 3 or PathSegment(particle[-1]).status == -1:
                raise Exception("Invalid trace")

            # Get state of particle initially
            p_first = PathSegment(particle[0])
            p_last = None

            # Determine final state of particle
            for i in range(len(particle) - 1, 1, -1):
                p_i = PathSegment(particle[i])
                p_j = PathSegment(particle[i - 1])
                if p_i.r > radius > p_j.r:
                    p_last = p_i
                    break

            if p_last is None:
                raise Exception("Invalid trace")

            # Return the particle's momentum and initial and final pixels
            initial_pixel = hp.vec2pix(nside, p_first.px, p_first.py,
                                       p_first.pz)
            final_pixel = hp.vec2pix(nside, p_last.px, p_last.py, p_last.pz)
            #b = (p_last.Bx, p_last.By, p_last.Bz)
            data_array.append((initial_pixel, final_pixel, p_last.p, p_last.Bx,p_last.By, p_last.Bz))

        except Exception:
            continue

    return data_array


# Create maps
def create_maps(nside, bins, obs_parameters, imposed_parameters,
                physical_index, particle_dir, particle_file,
                type="standard"):

    # Find particle data and load it
    particles_data = np.load(particle_dir + particle_file, allow_pickle=True)
    particles = particles_data["particles"]

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
    reweighed_initial = np.zeros((bins, npix))
    reweighed_final = np.zeros((bins, npix))

    # Populate initial and final maps
    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = -1
        for i in range(bins):
            if p >= bin_sizes[i]:
                p_bin += 1
            else:
                break
        particle_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1],
                                          physical_index, -1)
        particle_weight *= observational_weight(p, obs_parameters)
        final_maps[p_bin][final_pixel] += particle_weight

    # Go back through the data and reweigh the initial map
    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        p = item[2]
        b = item[3]
        p_bin = -1
        for i in range(bins):
            if p >= bin_sizes[i]:
                p_bin += 1
            else:
                break

        uniform = imposed_parameters[0]
        dipole = imposed_parameters[1]

        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, b)
        direction_weight = final_maps[p_bin][final_pixel]
        momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1],
                                          physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)

        total_weight = momentum_weight * imposed_weight * obs_weight \
            / direction_weight

        if type == "unweighted":
            total_weight = 1

        reweighed_initial[p_bin][initial_pixel] += total_weight
        reweighed_final[p_bin][final_pixel] += total_weight

    return np.array([reweighed_initial, reweighed_final])


# Create particle weighting file for KS tests
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
    reweighed_particles = [[] for i in range(npix)]

    # Populate initial and final maps
    for item in particles:
        final_pixel = int(item[1])
        p = item[2]
        p_bin = -1
        for i in range(bins):
            if p >= bin_sizes[i]:
                p_bin += 1
            else:
                break
        particle_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1],
                                          physical_index, -1)
        particle_weight *= observational_weight(p, obs_parameters)
        final_maps[p_bin][final_pixel] += particle_weight

    # Go back through the data and reweigh the initial map. Save the individual
    # particle data for statistical testing
    for item in particles:
        initial_pixel = int(item[0])
        final_pixel = int(item[1])
        p = item[2]
        b = item[3]
        p_bin = -1
        for i in range(bins):
            if p >= bin_sizes[i]:
                p_bin += 1
            else:
                break

        uniform = imposed_parameters[0]
        dipole = imposed_parameters[1]

        imposed_weight = uniform + dipole * cos_dipole_f(nside, final_pixel, b)
        direction_weight = final_maps[p_bin][final_pixel]
        momentum_weight = weight_powerlaw(p, bin_sizes[0], bin_sizes[-1],
                                          physical_index, -1)
        obs_weight = observational_weight(p, obs_parameters)

        reweighed_particles[initial_pixel].append([p, momentum_weight *
                                                  imposed_weight * obs_weight /
                                                  direction_weight])
    return reweighed_particles


# For applying a dipole to the final distribution of momenta
def cos_dipole_f(nside, pix, b):
    pxf, pyf, pzf = hp.pix2vec(nside, pix)
    return -(pxf * b[0] + pyf * b[1] + pzf * b[2]) / \
        (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1.e-16) / \
        np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])


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


# Rotate simulation coordinates
def rotate_map_sim(old_map):
    map_matrix = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    npix = len(old_map)
    nside = hp.npix2nside(npix)
    new_map = np.zeros(npix)

    # For each pixel in the new map, add the transformed pixel from the old map
    for i in range(npix):
        theta, phi = hp.pix2ang(nside, i)

        # Apply transform from simulation to ecliptic coordinates
        old_theta, old_phi = hp.rotator.rotateDirection(
            np.linalg.inv(map_matrix), theta, phi)

        # Add appropriate pixel to new map
        old_pix = hp.ang2pix(nside, old_theta, old_phi)
        new_map[i] += old_map[old_pix]

    return new_map


# Create a log spaced energy binning scheme
def create_bin_sizes(particles, num_bins):
    max_energy = 0
    min_energy = sys.float_info.max
    for pixel in particles:
        for particle in pixel:
            if particle[0] > max_energy:
                max_energy = particle[0]
            if particle[0] < min_energy:
                min_energy = particle[0]
    max_log = np.log10(1.001 * max_energy)
    min_log = np.log10(0.99 * min_energy)
    cutoffs = np.logspace(min_log, max_log, num=num_bins + 1, base=10)
    return cutoffs


# Sort particles into energy bins. Returns a 3D array with pixels on axis 0,
# bins on axis 1 and particles on axis 2
def bin_particles(pixels, binning):
    num_bins = len(binning) - 1
    pixels_binned = []
    for particles_list in pixels:
        particles_binned = [[] for i in range(num_bins)]
        for particle in particles_list:
            for i in range(num_bins):
                if binning[i] < particle[0] < binning[i + 1]:
                    particles_binned[i].append(particle)
                    break
        pixels_binned.append(particles_binned)
    return pixels_binned


# Create sky map of reweighed momenta
def create_reweighed_sky_maps(binned_particles):
    num_pixels = len(binned_particles)
    num_bins = len(binned_particles[0])

    flux_maps = np.zeros((num_bins, num_pixels))

    for i in range(num_bins):
        for j in range(num_pixels):
            for particle in binned_particles[j][i]:
                flux_maps[i][j] += particle[1]
    return flux_maps


# Gaussian in log space for approximating experimental sensitivity
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
