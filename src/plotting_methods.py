import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.stats as stat
from matplotlib import pylab
from tqdm import tqdm
from scipy.stats import chi2
from statistical_methods import get_ring_distribution, impose_energy_range


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

def plot_chi_squared(chi_squared_map, out_dir, name):
    chi2sum = chi_squared_map
    # print('np.argmax(chi2sum) ' + name, np.argmax(chi2sum))
    # print('np.argmin(chi2sum) ' + name, np.argmin(chi2sum))
    print('chi2sum[np.argmax(chi2sum)] ' + name, chi2sum[np.argmax(chi2sum)])
    print('chi2sum[np.argmin(chi2sum)] ' + name, chi2sum[np.argmin(chi2sum)])
    #print('chi2sum',chi2sum)
    #signs = np.sign(p_values)
    #print('signs',signs)
    z_values = np.abs(chi2sum)
    #z_values = np.maximum(-stat.norm.ppf(np.abs(p_values)), 0) * signs
    #z_values = np.maximum(-stat.norm.ppf(np.abs(np.nan_to_num(p_values))), 0) * np.nan_to_num(signs)
    plot_skymap(z_values,
                title=name,
                label="Range",
                proj='C0',
                dMin=5.0,
                dMax=1020.0,
                filename=out_dir + name)
    plt.close()

def plot_histogram(pix_ind, map, limits, width, out_dir):
    # Physical constants for scaling momentum
    c = 299792458
    e = 1.60217663 * 10 ** (-19)
    m_p = 1.67262192 * 10 ** (-27)
    npix = len(map)
    nside = hp.npix2nside(npix)
    lower = limits[0] / (m_p * c * c / (e * 10 ** 12))
    upper = limits[1] / (m_p * c * c / (e * 10 ** 12))

    # for i in tqdm(range(npix)):
    #     pixel_distribution_v2, npix_on_v2 = get_ring_distribution(i, map, nside, width)
    #     pixel_distribution_v2 = impose_energy_range(pixel_distribution_v2, lower, upper)
    # print('ind_max', np.argmax(pixel_distribution_v2[0]))
    # print('ind_min', np.argmin(pixel_distribution_v2[0]))
    # print('max_value', pixel_distribution_v2[0][np.argmax(pixel_distribution_v2[0])])
    # print('min_value', pixel_distribution_v2[0][np.argmin(pixel_distribution_v2[0])])

    pixel_distribution, npix_on = get_ring_distribution(pix_ind, map, nside, width)
    pixel_distribution = impose_energy_range(pixel_distribution, lower, upper)
    ebins = np.logspace(2.5, 5.0, 20)
    plt.hist(pixel_distribution[0], bins = ebins, edgecolor='black', histtype='step', label='Energy histogram pixel ' + str(pix_ind))
    plt.loglog()
    plt.legend()
    plt.xlabel('Energy bins')
    plt.title('Histogram')
    fig = pylab.figure(1)
    fig.savefig(out_dir + 'hist_on_pixel_' + str(pix_ind) + '.png', dpi=250)
    plt.close()

def plot_histogram_whole_skymap(map, out_dir, name):
    ebins = np.logspace(1.0, 5.0, 20)
    plt.hist(map, bins = ebins, histtype='step', label='Chi2sum whole skymap ' + name)
    plt.loglog()
    plt.legend()
    plt.xlabel('Energy bins')
    plt.title('Histogram')
    fig = pylab.figure(1)
    fig.savefig(out_dir + name + '.png', dpi=250)
    plt.close()

def plot_histogram_chi2sum(chi2sum, out_dir, name):
    ebins = np.logspace(0.0, 4.0, 11)
    # Create histogram of chi-squared/ndf values
    plt.hist(chi2sum, bins=ebins, color='blue', density=False, alpha=0.5, histtype='step', linewidth=1.5)
    df = len(ebins) - 1
    # Plot the chi-squared distribution function for the corresponding degrees of freedom
    x = np.logspace(0.0, 4.0, 1000)  # Adjust range as needed
    y = chi2.pdf(x, df) * max(chi2sum)  # Scaling by df to get chi-squared distribution
    plt.plot(x, y, 'r-', lw=2, label=f'Chi-squared ({df} df)')
    #plt.loglog()
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$\chi^2$ / ndf')
    plt.ylabel('Frequency')
    plt.title('Histogram of $\chi^2$/ndf')
    plt.grid(True)
    fig = pylab.figure(1)
    fig.savefig(out_dir + name + '.png', dpi=250)
    plt.close()

def plot_power(flux_map, out_dir, bin):
    spectrum = hp.anafast(flux_map, lmax=30)[1:]
    plt.scatter(range(1, len(spectrum) + 1), np.log10(spectrum))
    plt.savefig(out_dir + f'bin={bin}')
