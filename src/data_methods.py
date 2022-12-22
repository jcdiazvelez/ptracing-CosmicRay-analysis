import sys
import numpy as np
import healpy as hp

from PathSegment import PathSegment


def create_position_maps(file, nside, matrix):
    print("Processing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            # Get particle track
            datum = data1[key]

            # Remove particles with insufficient crossings
            if len(datum) < 3:
                break

            # Get state of particle initially and at r = 330 au
            p_first = PathSegment(datum[0])
            p_last = PathSegment(datum[-1])

            # Use only particles which terminate at 55000 au
            if p_last.r < 50000:
                break

            # Remove particles which failed
            if p_last.status < 0:
                break

            initial_momentum = hp.rotator.rotateVector(matrix, p_first.px, p_first.py, p_first.pz)
            final_momentum = hp.rotator.rotateVector(matrix, p_last.px, p_last.py, p_last.pz)

            # Determine initial and final pixels, returning these and the particle's momentum
            initial_pixel = hp.vec2pix(nside, initial_momentum[0], initial_momentum[1], initial_momentum[2])
            final_pixel = hp.vec2pix(nside, final_momentum[0], final_momentum[1], final_momentum[2])
            data_array.append((initial_pixel, final_pixel, p_last.p))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array


def powerlaw_pdf(x, x_min, x_max, g):
    x_min_g, x_max_g = x_min ** (g + 1.), x_max ** (g + 1.)
    if g == -1.0:
        return x ** g / np.log(x_max / x_min)
    else:
        return (g + 1.) / (x_max_g - x_min_g) * x ** g


# Weighting scheme for energy bins
def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)
