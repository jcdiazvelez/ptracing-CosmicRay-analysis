import sys
import numpy as np
import healpy as hp

from PathSegment import PathSegment


def create_position_maps(file, nside):
    print("Processing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            datum = data1[key]                      # Get track

            if len(datum) < 3:
                break

            p_first = PathSegment(datum[0])
            p_last = PathSegment(datum[-2])
            if p_last.status < 0:
                break                               # If particle failed, don't include it in data

            data_array.append((hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz),
                               hp.vec2pix(nside, p_last.x, p_last.y, p_last.z), p_last.p))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array


def perform_weighting(file, nside, final_maps, bin_sizes):
    print("Reweighing: " + file)
    data1 = np.load(file)

    data_array = []

    for key in data1:
        try:
            datum = data1[key]                                           # Get track

            if len(datum) < 3:
                break

            p_first = PathSegment(datum[0])                              # Get first and last positions of particle
            p_last = PathSegment(datum[-2])
            if p_last.status < 0:
                break                                                    # If particle failed, don't include it in data

            p = p_first.p
            p_bin = 0
            for i in range(len(bin_sizes)):
                if p >= bin_sizes[i]:
                    p_bin += 1

            initial_pixel = hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz)
            final_pixel = hp.vec2pix(nside, p_last.x, p_last.y, p_last.z)
            weight = 1.0 / final_maps[p_bin][final_pixel]

            data_array.append((initial_pixel, weight, p_bin))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array
