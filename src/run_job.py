import json
import os
import numpy as np

from data_methods import create_particles, create_weights, create_maps, \
    create_kolmogorov_map

# Import configuration data for the job, and set up useful variables

with open("config.json") as config_file:
    job_data = json.load(config_file)

# First determine whether new particle data needs to be created. If
# data needs to be created, create it and save it

nside = job_data["nside"]

particle_dir = job_data["particle_data_location"]
particle_file = f"particles_nside={nside}.npz"

if not os.path.exists(particle_dir + particle_file):
    create_particles(nside, particle_dir, particle_file)

# Config parameters to determine which weightings we need to produce

binnings = job_data["binnings"]
imposed_parameters = job_data["imposed_distribution"]
use_observational = job_data["observational?"]
obs_parameters = job_data["observational_parameters"]
run_kolmogorov = job_data["kolmogorov?"]
run_unweighted = job_data["plot_unweighted?"]
run_imposed = job_data["plot_imposed?"]
physical_index = job_data["physical_index"]
weights_dir = job_data["weighted_data_location"]

if run_kolmogorov:
    binnings.append(50)

# Produce all required weights files
if not use_observational:
    obs_parameters[0] = -1
    obs_parameters[1] = -1

for bins in binnings:
    weights_file = f"weights_nside={nside}_" + \
                    f"bins={bins}_" + \
                    f"obs={int(100 * obs_parameters[0])}+" + \
                    f"{int(obs_parameters[1])}_" + \
                    f"distribution={int(10000 * imposed_parameters[0])}+" + \
                    f"{int(10000 * imposed_parameters[1])}_" + \
                    f"unweighted={int(run_unweighted)}_" + \
                    f"imposed={int(run_imposed)}_" + \
                    f"phys_index={int(100 * physical_index)}.npz"

    # Determine if weights file exists, and generate it if it doesn't
    if not os.path.exists(weights_dir + weights_file):
        create_weights(nside, bins, obs_parameters, imposed_parameters,
                       run_unweighted, run_imposed, physical_index,
                       particle_dir, particle_file,
                       weights_dir, weights_file)

# Create required sky maps from weight files
job_dir = job_data["map_data_location"]
if not os.path.exists(job_dir):
    os.makedirs(job_dir)

# For each binning
for bins in binnings:
    weights_file = f"weights_nside={nside}_" + \
                    f"bins={bins}_" + \
                    f"obs={int(100 * obs_parameters[0])}+" + \
                    f"{int(obs_parameters[1])}_" + \
                    f"distribution={int(10000 * imposed_parameters[0])}+" + \
                    f"{int(10000 * imposed_parameters[1])}_" + \
                    f"unweighted={int(run_unweighted)}_" + \
                    f"imposed={int(run_imposed)}_" + \
                    f"phys_index={int(100 * physical_index)}.npz"

    # Read in weights file
    with np.load(weights_dir + weights_file, allow_pickle=True) as weight_data:

        # Create initial and final reweighed maps
        initial_data = weight_data["initial"]
        initial_maps = create_maps(initial_data)
        final_data = weight_data["final"]
        final_maps = create_maps(final_data)

        # Create unweighted maps if asked
        initial_data_unweighted = weight_data["unweighted_initial"]
        initial_unweighted = create_maps(initial_data_unweighted,
                                         run=run_unweighted)
        final_data_unweighted = weight_data["unweighted_final"]
        final_unweighted = create_maps(final_data_unweighted,
                                       run=run_unweighted)

        # Create imposed maps if asked
        initial_data_imposed = weight_data["imposed_initial"]
        initial_imposed = create_maps(initial_data_imposed, run=run_imposed)
        final_data_imposed = weight_data["imposed_final"]
        final_imposed = create_maps(final_data_imposed, run=run_imposed)

    # Save binned maps in a new file
    np.savez_compressed(job_dir + f"bins={bins}",
                        initial=initial_maps,
                        final=final_maps,
                        initial_unweighted=initial_unweighted,
                        final_unweighted=final_unweighted,
                        initial_imposed=initial_imposed,
                        final_imposed=final_imposed)

# Create Kolmogorov maps if asked
if run_kolmogorov:
    weights_file = f"weights_nside={nside}_" + \
                    f"bins={50}_" + \
                    f"obs={int(100 * obs_parameters[0])}+" + \
                    f"{int(obs_parameters[1])}_" + \
                    f"distribution={int(10000 * imposed_parameters[0])}+" + \
                    f"{int(10000 * imposed_parameters[1])}_" + \
                    f"unweighted={int(run_unweighted)}_" + \
                    f"imposed={int(run_imposed)}_" + \
                    f"phys_index={int(100 * physical_index)}.npz"

    with np.load(weights_dir + weights_file, allow_pickle=True) as weight_data:
        width = job_data["kolmogorov_width"]
        limits = job_data["kolmogorov_limits"]
        kolmogorov_data = weight_data["initial"]
        kolmogorov_map = create_kolmogorov_map(kolmogorov_data, width, limits,
                                               use_observational)

    # Save Kolmogorov maps in a new file
    np.savez_compressed(job_dir + "kolmogorov",
                        kolmogorov=kolmogorov_map)

# Plot all created sky maps

# Plot binned maps

# Plot Kolmogorov maps
