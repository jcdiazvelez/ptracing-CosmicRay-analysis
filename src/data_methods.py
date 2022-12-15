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
            # Get particle track
            datum = data1[key]

            # Remove particles with insufficient crossings
            if len(datum) < 3:
                break

            # Get state of particle initially and at r = 330 au
            p_first = PathSegment(datum[0])
            p_last = PathSegment(datum[-2])

            # Remove particles which failed
            if p_last.status < 0:
                break

            # Determine initial and final pixels, returning these and the particle's momentum
            initial_pixel = hp.vec2pix(nside, p_first.px, p_first.py, p_first.pz)
            final_pixel = hp.vec2pix(nside, p_last.px, p_last.py, p_last.pz)
            data_array.append((initial_pixel, final_pixel, p_last.p))

        except:
            sys.excepthook(*sys.exc_info())
            break

    return data_array
