import json
import os
import numpy as np

from data_methods import create_particles, create_maps, \
    create_weights
from statistical_methods import perform_kolmogorov_smirnov

# Import configuration data for the job, and set up useful variables

with open("config.json") as config_file:
    job_data = json.load(config_file)

# First determine whether new particle data needs to be created. If
# data needs to be created, create it and save it

nside = job_data["nside"]

particle_dir = job_data["particle_data_location"]
particle_file = f"particles_nside={nside}.npz"

if not os.path.exists(particle_dir + particle_file):
    raw_dir = job_data["raw_data_location"]
    create_particles(nside, particle_dir, particle_file, raw_dir)

# Config parameters to determine which weightings we need to produce

binnings = job_data["binnings"]
imposed_parameters = job_data["imposed_distribution"]
use_observational = job_data["observational?"]
obs_parameters = job_data["observational_parameters"]
run_kolmogorov = job_data["kolmogorov?"]
run_unweighted = job_data["plot_unweighted?"]
physical_index = job_data["physical_index"]
maps_dir = job_data["map_data_location"]

# Set observational parameters to null values if observational profile
# is not being used
if not use_observational:
    obs_parameters[0] = -1
    obs_parameters[1] = -1

# Produce all required weights files
for bins in binnings:
    # Create necessary sky maps for the binning
    standard_maps = create_maps(nside, bins, obs_parameters,
                                imposed_parameters, physical_index,
                                particle_dir, particle_file)
    np.savez_compressed(maps_dir + f"standard_bins={bins}.npz")

    if run_unweighted:
        unweighted_maps = create_maps(nside, bins, obs_parameters,
                                      imposed_parameters, physical_index,
                                      particle_dir, particle_file,
                                      type="unweighted")
        np.savez_compressed(maps_dir + f"unweighted={bins}.npz")

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

    # Save Kolmogorov map in a new file
    np.savez_compressed(maps_dir + "kolmogorov.npz")
