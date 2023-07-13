import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.stats as stat
from matplotlib import pylab


def plot_skymap(skymap, title, proj='C', label='', filename=None, thresh=None,
                dMin=None, dMax=None, sun=None):
    params = {'legend.fontsize': 'x-large',
              # 'axes.labelsize': 'x-large',
              'axes.titlesize': '20',
              # 'xtick.labelsize': 'x-large',
              # 'ytick.labelsize': 'x-large',
              "font.family": "serif"}
    pylab.rcParams.update(params)

    colormap = pylab.get_cmap("coolwarm")
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


def plot_flux(flux_map, out_dir, plot_type, bin):
    smoothed = hp.sphtfunc.smoothing(flux_map, fwhm=0.1)
    if plot_type == "initial":
        location = "Earth"
    else:
        location = "Simulation Boundary"

    plot_skymap(smoothed,
                title=f"Flux Distribution at {location} for Bin {bin}",
                label="Relative Flux",
                proj='C0',
                # dMin=-0.0013,
                # dMax=0.0013,
                filename=out_dir + plot_type + f"bin={bin}")
    plt.close()


def plot_kolmogorov(ks_map, out_dir):
    p_values = ks_map
    signs = np.sign(p_values)
    z_values = np.maximum(-stat.norm.ppf(np.abs(p_values)), 0) * signs
    plot_skymap(z_values,
                title='Kolmogorov-Smirnov Z-Scores',
                label="Significance / $\sigma$",
                proj='C0',
                dMin=-12,
                dMax=12,
                filename=out_dir + 'kolmogorov')
    plt.close()
