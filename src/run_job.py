import json
import os
import numpy as np

from data_methods import create_particles, create_maps, create_weights, rotate_map
from statistical_methods import perform_kolmogorov_smirnov, perform_chi_squared

# Import configuration data for the job, and set up useful variables

with open("config.json") as config_file:
    job_data = json.load(config_file)

# First determine whether new particle data needs to be created. If
# data needs to be created, create it and save it

nside = job_data["nside"]

particle_dir = job_data["particle_data_location"]
particle_file = f"nside={nside}.npz"

if not os.path.exists(particle_dir + particle_file):
    raw_dir = job_data["raw_data_location"]
    #create_particles(nside, particle_dir, particle_file, raw_dir)

# Config parameters to determine which weightings we need to produce

binnings = job_data["binnings"]
imposed_parameters = job_data["imposed_distribution"]
use_observational = job_data["observational?"]
obs_parameters = job_data["observational_parameters"]
run_kolmogorov = job_data["kolmogorov?"]
run_chi_squared = job_data["chi_squared?"]
run_hist_on_pixel = job_data["hist_on_pixel?"]
run_unweighted = job_data["plot_unweighted?"]
physical_index = job_data["physical_index"]
maps_dir = job_data["map_data_location"]

# Set observational parameters to null values if observational profile
# is not being used
if not use_observational:
    obs_parameters[0] = -1
    obs_parameters[1] = -1

# Produce all required weights files
# for bins in binnings:
#     # Create necessary sky maps for the binning
#     standard_maps = create_maps(nside, bins, obs_parameters,
#                                 imposed_parameters, physical_index,
#                                 particle_dir, particle_file)
#     # Rotate maps to appropriate coordinate system
#     for i in range(bins):
#         standard_maps[0][i] = rotate_map(standard_maps[0][i])
#         standard_maps[1][i] = rotate_map(standard_maps[1][i])
#     np.savez_compressed(maps_dir + f"standard_bins={bins}.npz",
#                         flux=standard_maps)

#     if run_unweighted:
#         unweighted_maps = create_maps(nside, bins, obs_parameters,
#                                       imposed_parameters, physical_index,
#                                       particle_dir, particle_file,
#                                       type="unweighted")
#         unweighted_maps[0][i] = rotate_map(unweighted_maps[0][i])
#         unweighted_maps[1][i] = rotate_map(unweighted_maps[1][i])
#         np.savez_compressed(maps_dir + f"unweighted={bins}.npz",
#                             flux=unweighted_maps)

# Create Kolmogorov maps if asked
if run_kolmogorov:
    width = job_data["kolmogorov_width"]
    limits = job_data["kolmogorov_limits"]
    kolmogorov_particles = create_weights(nside, 50, obs_parameters,
                                          imposed_parameters, physical_index,
                                          particle_dir, particle_file)
    kolmogorov_particles = np.array(kolmogorov_particles)
    kolmogorov_map = perform_kolmogorov_smirnov(kolmogorov_particles, limits,
                                                width)
    kolmogorov_map = rotate_map(kolmogorov_map)
    np.savez_compressed(maps_dir + "kolmogorov.npz",
                        kolmogorov=kolmogorov_map)

# Create Chi-squared maps if asked
# if run_chi_squared:
#     width = job_data["chi_squared_width"]
#     limits = job_data["chi_squared_limits"]
#     weighted_particles = create_weights(nside, 50, obs_parameters,
#                                           imposed_parameters, physical_index,
#                                           particle_dir, particle_file)
#     weighted_particles = np.array(weighted_particles)
#     chi_squared_map = perform_chi_squared(weighted_particles, limits,
#                                                 width)
    # chi_squared_map = rotate_map(chi_squared_map)
    # np.savez_compressed(maps_dir + "chi_squared_updated.npz",
    #                     chi_squared=chi_squared_map)

if run_chi_squared:
    width = job_data["chi_squared_width"]
    limits = job_data["chi_squared_limits"]
    weighted_particles = create_weights(nside, 50, obs_parameters,
                                          imposed_parameters, physical_index,
                                          particle_dir, particle_file)
    weighted_particles = np.array(weighted_particles)
    chi_squared_map = perform_chi_squared(weighted_particles, limits,
                                                width)
    chi_squared_map = rotate_map(chi_squared_map)
    np.savez_compressed(maps_dir + "chi_squared_april3.npz",
                        chi_squared=chi_squared_map)

# if run_hist_on_pixel:
#     width = job_data["hist_on_pixel_width"]
#     limits = job_data["hist_on_pixel_limits"]
#     weighted_particles = create_weights(nside, 50, obs_parameters,
#                                           imposed_parameters, physical_index,
#                                           particle_dir, particle_file)
#     weighted_particles = np.array(weighted_particles)
#     weighted_particles = rotate_map(weighted_particles)
#     np.savez_compressed(maps_dir + "weighted_particles.npz",
#                         weighted_particles_map=weighted_particles)
    #npix_on_map = perform_histogram_on_pixel(1000, weighted_particles, limits, width)