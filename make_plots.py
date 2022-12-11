import glob

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

filename = "a_crossings_ptracing_energy_.npz"
path = "./" + filename

data = np.load(filename)

for key in data:
    hp.visufunc.mollview(data[key])
    plt.title(key)
    plt.savefig('./figs/' + key)

