import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

filename = "a_crossings_c_energy_.npz"
path = "../data/" + filename

data = np.load(filename)

for key in data:
    sky_map = hp.remove_monopole(data[key])
    power_spectrum = hp.anafast(sky_map)
    hp.visufunc.mollview(sky_map)
    plt.title(key)
    plt.savefig('../figs/' + key)
    plt.clf()
    plt.plot(np.log(power_spectrum))
    plt.savefig('../figs/' + key + 'power')
