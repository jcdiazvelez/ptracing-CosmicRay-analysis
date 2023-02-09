import sys
import numpy as np
import healpy as hp

from PathSegment import PathSegment


# Process data files and populate initial and final sky maps
def process_particle_data(filename, nside, radius):
    print("Processing: " + filename)
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

            # Determine initial and final pixels, returning these and the particle's momentum
            initial_pixel = hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz)
            final_pixel = hp.vec2pix(nside, p_last.px, p_last.py, p_last.pz)
            data_array.append((initial_pixel, final_pixel, p_last.p, p_last.time))

        except:
            continue

    return data_array


# For applying a dipole to the final distribution of momenta
def cos_dipole_f(nside, pix, bx=-1.737776, by=-1.287260, bz=2.345265):
    pxf, pyf, pzf = hp.pix2vec(nside, pix)
    return -(pxf * bx + pyf * by + pzf * bz) / (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1.e-16) / np.sqrt(
        bx * bx + by * by + bz * bz)


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
    coord_matrix = np.matrix([[-0.202372670869508942, 0.971639226673224665, 0.122321361599999998],
                               [-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],
                               [-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]])
    map_matrix = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    npix = len(old_map)
    nside = hp.npix2nside(npix)
    new_map = np.zeros(npix)
    r = hp.Rotator(coord=['C', 'E'])

    # For each pixel in the new map, add the transformed pixel from the old map
    for i in range(npix):
        theta, phi = hp.pix2ang(nside, i)

        # Apply transform from simulation to ecliptic coordinates
        old_theta, old_phi = hp.rotator.rotateDirection(np.linalg.inv(map_matrix), theta, phi)

        # Appy transform from ecliptic to equatorial coordinates
        old_theta, old_phi = r(old_theta, old_phi)

        # Apply transform to put GMT on rhs of maps
        old_theta, old_phi = hp.rotator.rotateDirection(np.linalg.inv(coord_matrix), old_theta, old_phi)

        # Add appropriate pixel to new map
        old_pix = hp.ang2pix(nside, old_theta, old_phi)
        new_map[i] += old_map[old_pix]

    return new_map


# Sort particles into energy bins
def bin_particles(particles, binning):
    return 0


# Create sky map of reweighed momenta
def create_reweighed_sky_maps(binned_particles):
    reweighed_maps = np.zeros(len(binned_particles))

    return reweighed_maps


# Create sky map of the average time of particles to propagate out of the system
def create_time_maps(binned_particles):
    timed_maps = np.zeros(len(binned_particles))

    return timed_maps
