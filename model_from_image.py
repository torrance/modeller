#! /usr/bin/env python
from __future__ import print_function, division

import argparse
import math
import sys
import time as tm

from astrobits.coordinates import radec_to_lm, fits_to_radec
from astrobits.mwabeam import MWABeam
from astropy.io import fits
from astropy.time import Time
from casacore.tables import table, taql
from numba import njit, prange, cuda, float32, float64, complex64, complex128
import numpy as np

from astrobits.mwabeam import MWABeam
from modeller.predict import predict


def main(args):
    mwabeam = MWABeam(args.metafits)

    mset = table(args.mset, readonly=False)
    freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
    midfreq = np.mean(freqs)

    # Reset data
    data = mset.getcol('DATA')
    data[:] = 0
    mset.putcol('DATA', data)

    print("Calculating pixel coordinates...")
    hdu = fits.open(args.modelimage)[0]
    model = hdu.data.flatten()
    ras, decs = fits_to_radec(hdu)  # each is a 4d array
    ras, decs = ras.flatten(), decs.flatten()
    print("Done")

    # Filter out all zero flux elements
    zeros = (model == 0)
    model = model[~zeros]
    ras, decs = ras[~zeros], decs[~zeros]

    # # Limit to 1000 objects
    # limit = 1000
    # model = model[0:limit]
    # ras, decs = ras[0:limit].copy(), decs[0:limit].copy()

    # Use spectral index -1 based on midfreq
    # s(nu) = A * nu^alpha
    # => A = s / nu^alpha

    batch = 1000
    for start in range(0, len(model), batch):
        end = start + batch
        print("Predicting %d-%d / %d model components" % (start, start + len(model[start:end]), len(model)))

        # A = model / midfreq**-1
        # model = A[start:end, None] * freqs[None, :]**-1
        modelbatch = model[start:end, None] * np.ones_like(freqs)

        predict(mset, mwabeam, ras[start:end], decs[start:end], modelbatch, applybeam=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mset', required=True)
    parser.add_argument('--modelimage', required=True)
    parser.add_argument('--metafits', required=True)
    args = parser.parse_args()
    main(args)
