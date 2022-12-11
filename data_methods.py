import sys
import numpy as np
import healpy as hp

from PathSegment import PathSegment


def run_file(file, nside):
    print("Processing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            datum = data1[key]  # Get track

            if len(datum) < 3:
                break

            p_first = PathSegment(datum[0])
            p_last = PathSegment(datum[-2])
            if p_last.status < 0:
                break  # If particle failed, don't include it in data

            data_array.append((hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz) + 1, weight))
            data_array.append((-hp.vec2pix(nside, p_last.x, p_last.y, p_last.z) - 1, weight))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array


def reweigh_file(file, nside, weights, energy_falloff=2.7):
    print("Reweighing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            datum = data1[key]  # Get track

            if len(datum) < 3:
                break

            p_first = PathSegment(datum[0])
            p_last = PathSegment(datum[-2])
            if p_last.status < 0:
                break  # If particle failed, don't include it in data

            weight = np.power(p_first.p / 1000, energy_falloff - 1)

            pix = hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz)
            fin_pix = hp.vec2pix(nside, p_last.x, p_last.y, p_last.z)
            data_array.append((pix, weights[fin_pix] * weight))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array
