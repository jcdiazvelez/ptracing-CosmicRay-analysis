import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.stats as stat
from argparse import ArgumentParser

from matplotlib import pylab


def plot_skymap(skymap, title, proj='C', label='', filename=None, thresh=None, dMin=None, dMax=None, sun=None):
    params = {'legend.fontsize': 'x-large',
              # 'axes.labelsize': 'x-large',
              'axes.titlesize': '20',
              # 'xtick.labelsize': 'x-large',
              # 'ytick.labelsize': 'x-large',
              "font.family": "serif"}
    pylab.rcParams.update(params)

    colormap = pylab.get_cmap("coolwarm")
    # colormap = pylab.get_cmap("jet")
    colormap.set_under("w")

    if proj == 'C0':
        rotation = (0, 0, 0)
    elif proj == 'C':
        rotation = (-180, 0, 0)
    else:
        rotation = proj

    hp.mollview(skymap,
                fig=1,
                title=title,
                rot=rotation,  # coord=['C'],
                unit=label,
                margins=(0.0, 0.03, 0.0, 0.13),
                notext=False, cmap=colormap, min=dMin, max=dMax)

    fig = pylab.figure(1)
    for ax in fig.get_axes()[0:1]:
        if proj == 'C0':
            ax.annotate("0$^\circ$", xy=(1.8, 0.625), size="x-large")
            ax.annotate("360$^\circ$", xy=(-1.95, 0.625), size="x-large")
        elif proj == 'C':
            ax.annotate("0$^\circ$", xy=(1.8, 0.625), size="x-large")
            ax.annotate("360$^\circ$", xy=(-1.95, 0.625), size="x-large")

    if sun is not None:
        hp.projscatter(sun[0], sun[1], lonlat=True, coord='C')
        hp.projtext(sun[0], sun[1], 'Sun', lonlat=True, coord='C', fontsize=18)

    hp.graticule()

    if filename:
        fig.savefig(filename, dpi=250)


# Parser for reading command line arguments
parser = ArgumentParser()
parser.add_argument("-N", "--nside", type=int, default='5')
parser.add_argument("-p", "--path", type=str, default='../figs/')
parser.add_argument("-o", "--outdir", type=str, default='../figs/')

args = parser.parse_args()
args_dict = vars(args)

# Read in data file
filename = args.path + 'nside=' + str(args.nside) + '.npz'
data = np.load(filename, allow_pickle=True)

# Read in data
flux_maps = data['flux']
flux_maps_final = data['flux_final']
kolmogorov_dist_maps = data['kolmogorov']
bin_limits = data['bin_limits']
limits = data['limits']
bins = data['bins']
widths = data['widths']

# Make directories for saving figures
out_path = args.outdir + f'nside={args.nside + 10000}/'

# Physical constants for scaling energy
c = 299792458
e = 1.60217663 * 10 ** (-19)
m_p = 1.67262192 * 10 ** (-27)
energy_factor = 1 / (m_p * c * c / (e * 10 ** 12))

# Plot binned tests
bin_counter = 0
for i in range(len(bins)):
    binning = bin_limits[i]
    suffix = f'bins={bins[i]}/'
    os.makedirs(out_path + 'flux/' + suffix)
    os.makedirs(out_path + 'flux_final/' + suffix)

    for j in range(bins[i]):
        lower_limit = binning[j] / energy_factor
        upper_limit = binning[j + 1] / energy_factor

        plt.set_cmap('coolwarm')
        smoothed = hp.sphtfunc.smoothing(flux_maps[bin_counter], fwhm=0.1)
        plot_skymap(smoothed,
                    title=f"Dipole Initial Distribution \n" + "{0:.3g}".format(lower_limit) + " TeV < E < "
                                                                                              "{0:.3g}".format(
                        upper_limit) + " TeV",
                    label="Relative Flux",
                    proj='C0',
                    dMin=-0.0013,
                    dMax=0.0013,
                    filename=out_path + 'flux/' + suffix + f'flux_map_num_bins={bins[i]}_bin={j}')

        plt.close()

        plt.set_cmap('coolwarm')
        plot_skymap(flux_maps_final[bin_counter],
                    title=f"Imposed Dipole Distribution \n at the Simulation Boundary",
                    label="Flux",
                    proj='C0',
                    dMin=-0.001,
                    dMax=0.001,
                    filename=out_path + 'flux_final/' + suffix + f'flux_map_final_num_bins={bins[i]}_bin={j}')

        plt.close()

        bin_counter += 1

if len(kolmogorov_dist_maps) > 1:
    os.makedirs(out_path + 'kolmogorov_z/')
    limits_counter = 0
    for i in range(len(limits)):
        lower_limit = limits[i][0]
        upper_limit = limits[i][1]
        for j in range(len(widths)):
            width = widths[j]
            p_values = kolmogorov_dist_maps[limits_counter]

            signs = np.sign(p_values)
            z_values = np.maximum(-stat.norm.ppf(np.abs(p_values)), 0) * signs

            plot_skymap(z_values,
                        title='Kolmogorov-Smirnov Z-Scores for \n Dipole Initial Distribution',
                        label="Significance / $\sigma$",
                        proj='C0',
                        dMin=-12,
                        dMax=12,
                        filename=out_path + 'kolmogorov_z/' + f'kolmogorov_z_uniform_limit={i}_width={width}')

            plt.close()

        limits_counter += 1
