import json
import os
import glob
import numpy as np
from plotting_methods import plot_kolmogorov, plot_chi_squared, plot_histogram, plot_histogram_whole_skymap, plot_flux, plot_power, plot_histogram_chi2sum

# Read in JSON and determine maps directory
with open("config.json") as config_file:
    job_data = json.load(config_file)

maps_dir = job_data["map_data_location"]
figs_dir = job_data["output_location"]

# Create globbed files list
filename = "*.npz"
path = maps_dir + "/" + filename
map_files = sorted(glob.glob(path))

# Iterate through map files
for file in map_files:
    file_ID = file.rsplit('/', 1)[-1]  # Remove path
    file_ID = file_ID[:-4]  # Remove file extension

    # Create plot directory
    file_plot_dir = figs_dir + file_ID + '/'
    if not os.path.exists(file_plot_dir):
        os.makedirs(file_plot_dir)

    # Plot with appropriate parameters
    # if file_ID == "kolmogorov":
    #     ks_map_data = np.load(file, allow_pickle=True)
    #     ks_map = ks_map_data["kolmogorov"]
    #     plot_kolmogorov(ks_map, file_plot_dir)

    # if file_ID == "test_weights_phy_ind_minus1_final_pix":
    #     chi_squared_data = np.load(file, allow_pickle=True)
    #     chi_squared_map = chi_squared_data["test_weights"]
    #     plot_chi_squared(chi_squared_map, file_plot_dir, 'test_weights_phy_ind_minus1_final_pix')
    #     #plot_histogram_chi2sum(chi_squared_map, file_plot_dir, 'chi_squared_updated_hist')

    if file_ID == "test_wei_fix_chi2_3":
        chi_squared_data = np.load(file, allow_pickle=True)
        chi_squared_map = chi_squared_data["test_weights"]
        plot_chi_squared(chi_squared_map, file_plot_dir, 'test_wei_fix_chi2_3')
        #plot_histogram_chi2sum(chi_squared_map, file_plot_dir, 'chi_squared_updated_hist')

    #else:
        # flux_maps_data = np.load(file, allow_pickle=True)
        # flux_maps = flux_maps_data["flux"]

        # initial_dir = file_plot_dir + "initial/"
        # final_dir = file_plot_dir + "final/"
        # power_dir = file_plot_dir + "power/"

        # if not os.path.exists(initial_dir):
        #     os.makedirs(initial_dir)
        # if not os.path.exists(final_dir):
        #     os.makedirs(final_dir)
        # if not os.path.exists(power_dir):
        #     os.makedirs(power_dir)

        # # Plot initial and final maps
        # for i in range(len(flux_maps[0])):
        #     plot_flux(flux_maps[0][i], initial_dir, "initial", i)
        #     plot_flux(flux_maps[1][i], final_dir, "final", i)
        #     plot_power(flux_maps[1][i], power_dir, i)
