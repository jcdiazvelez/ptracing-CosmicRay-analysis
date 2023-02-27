import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat
from data_methods import rotate_map
from argparse import ArgumentParser

# Parser for reading command line arguments
parser = ArgumentParser()
parser.add_argument("-f", "--file", type=str, default='third_round/nside=19.npz')
parser.add_argument("-p", "--path", type=str, default='../figs/')
parser.add_argument("-o", "--outdir", type=str, default='../figs/')

counter = 27

args = parser.parse_args()
args_dict = vars(args)

# Read in data file
filename = args.path + args.file
data = np.load(filename)

flux_maps = 0
flux_maps_final = 0
time_maps = 0
kolmogorov_dist_maps = 0

bin_limits = 0
limits = 0
bins = 0
widths = 0

# Read in data from file
for key in data:
    if key == 'flux':
        flux_maps = data[key]
    elif key == 'flux_final':
        flux_maps_final = data[key]
    elif key == 'time':
        time_maps = data[key]
    elif key == 'kolmogorov':
        kolmogorov_dist_maps = data[key]
    elif key == 'binlimits':
        bin_limits = data[key]
    elif key == 'limits':
        limits = data[key]
    elif key == 'bins':
        bins = data[key]
    elif key == 'widths':
        widths = data[key]

# Rotate all maps
for i in range(len(flux_maps)):
    flux_maps[i] = rotate_map(flux_maps[i])

for i in range(len(flux_maps_final)):
    flux_maps_final[i] = rotate_map(flux_maps_final[i])

for i in range(len(time_maps)):
    time_maps[i] = rotate_map(time_maps[i])

for i in range(len(kolmogorov_dist_maps)):
    kolmogorov_dist_maps[i] = rotate_map(kolmogorov_dist_maps[i])

# Physical constants for scaling energy
c = 299792458
e = 1.60217663 * 10 ** (-19)
m_p = 1.67262192 * 10 ** (-27)
energy_factor = 1 / (m_p * c * c / (e * 10 ** 12))

# Plot binned tests
bin_counter = 0
for i in range(len(bins)):
    binning = bin_limits[i]
    for j in range(bins[i]):
        lower_limit = binning[j] / energy_factor
        upper_limit = binning[j + 1] / energy_factor

        plt.set_cmap('coolwarm')
        hp.visufunc.mollview(np.log10(flux_maps[bin_counter] / stat.gmean(flux_maps[bin_counter])),
                             title=f'Initial flux skymap for ' + "{0:.3g}".format(lower_limit) +
                                   ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                             unit="Flux",
                             max=np.log10(3),
                             min=np.log10(1 / 3))
        hp.graticule()
        plt.savefig(args.outdir + f'flux/flux_map_num_bins={bins[i]}_bin={j}')

        plt.set_cmap('coolwarm')
        hp.visufunc.mollview(np.log10(flux_maps_final[bin_counter] / stat.gmean(flux_maps_final[bin_counter])),
                             title=f'Final flux skymap for ' + "{0:.3g}".format(lower_limit) +
                                   ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                             unit="Flux",
                             max=np.log10(3),
                             min=np.log10(1 / 3))
        hp.graticule()
        plt.savefig(args.outdir + f'flux_final/flux_map_final_num_bins={bins[i]}_bin={j}')

        plt.set_cmap('coolwarm')
        hp.visufunc.mollview(np.log10(time_maps[bin_counter]),
                             title=f'Time skymap for ' + "{0:.3g}".format(lower_limit) +
                                   ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                             unit="Log(Time /s)",
                             min=np.min(np.log10(time_maps)),
                             max=np.max(np.log10(time_maps)))
        hp.graticule()
        plt.savefig(args.outdir + f'time/time_map_num_bins={bins[i]}_bin={j}')
        plt.clf()

        bin_counter += 1

limits_counter = 0
for i in range(len(limits)):
    lower_limit = limits[i][0]
    upper_limit = limits[i][1]
    for j in range(len(widths)):
        width = widths[j]
        p_values = kolmogorov_dist_maps[limits_counter]

        plt.set_cmap('bone')
        hp.visufunc.mollview(np.abs(p_values),
                             title=f'Kolmogorov-Smirnov P-Values for ' + "{0:.3g}".format(lower_limit) +
                                   ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                             unit="P")
        hp.graticule()
        plt.savefig(args.outdir + f'kolmogorov-p/kolmogorov_map_p_limit={i+counter}_width={width}')

        signs = np.sign(p_values)
        z_values = np.maximum(stat.norm.ppf(1 - np.abs(p_values)), 0) * signs

        plt.set_cmap('coolwarm')

        hp.visufunc.mollview(z_values,
                             title=f'Kolmogorov-Smirnov Z-Scores for ' + "{0:.3g}".format(lower_limit) +
                                   ' TeV < E < ' + "{0:.3g}".format(upper_limit) + ' TeV',
                             unit="Sigma",
                             min=-5,
                             max=5)
        hp.graticule()

        plt.savefig(args.outdir + f'kolmogorov-z/kolmogorov_map_z_limit={i+counter}_width={width}')

    limits_counter += 1
