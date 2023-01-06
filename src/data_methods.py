import sys
import numpy as np
import healpy as hp

from PathSegment import PathSegment


# Pass radius to function
# Last position is the latest instant where the particle passes outwards through that radius


def create_position_maps(file, nside, radius):
    print("Processing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            # Get particle track
            datum = data1[key]

            if len(datum) < 3 or PathSegment(datum[-1]).status == -1:
                raise Exception("Invalid trace")

            # Get state of particle initially
            p_first = PathSegment(datum[0])
            p_last = None

            for i in range(len(datum) - 1, 1, -1):
                p_i = PathSegment(datum[i])
                p_j = PathSegment(datum[i - 1])
                if p_i.r > radius > p_j.r:
                    p_last = p_i
                    break

            if p_last is None:
                raise Exception("Invalid trace")

            # Determine initial and final pixels, returning these and the particle's momentum
            initial_pixel = hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz)
            final_pixel = hp.vec2pix(nside, p_last.px, p_last.py, p_last.pz)
            data_array.append((initial_pixel, final_pixel, p_last.p))

        except:
            continue

    return data_array


def cos_dipole_f(nside, pix, bx=-1.737776, by=-1.287260, bz=2.345265):
    pxf, pyf, pzf = hp.pix2vec(nside, pix)
    return -(pxf * bx + pyf * by + pzf * bz) / (np.sqrt(pxf * pxf + pyf * pyf + pzf * pzf) + 1.e-16) / np.sqrt(
        bx * bx + by * by + bz * bz)


def powerlaw_pdf(x, x_min, x_max, g):
    x_min_g, x_max_g = x_min ** (g + 1.), x_max ** (g + 1.)
    if g == -1.0:
        return x ** g / np.log(x_max / x_min)
    else:
        return (g + 1.) / (x_max_g - x_min_g) * x ** g


# Weighting scheme for energy bins
def weight_powerlaw(x, x_min, x_max, g, power):
    return x ** g / powerlaw_pdf(x, x_min, x_max, power)
