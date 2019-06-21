#! /usr/bin/env python
from __future__ import print_function, division

import argparse
import time as tm

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
from casacore.tables import addImagingColumns, table, taql
import numpy as np

from astrobits.coordinates import radec_to_lm
from astrobits.mwabeam import MWABeam
import astrobits.skymodel as skymodel

from modeller.predict import predict


def main(args):

    if not args.nosimulate:
        mset = table(args.mset, readonly=False)
        freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
        midfreq = np.mean(freqs)

        with open(args.model) as f:
            models = np.array(skymodel.parse(f))

        comps = np.array([comp for model in models for comp in model.components if comp.flux(midfreq) > args.fluxthreshold])
        ras = np.array([comp.ra for comp in comps])
        decs = np.array([comp.dec for comp in comps])

        mwabeam = MWABeam(args.metafits)
        print("Simulating %d components" % len(comps))

        # Reset data
        data = mset.getcol('DATA')
        data[:] = 0
        mset.putcol('DATA', data)

        # Calculate fluxes for each source for each frequency
        fluxes = np.empty((len(comps), len(freqs)))
        for i, comp in enumerate(comps):
            fluxes[i] = comp.flux(freqs)

        # Batch sources
        batch = 100000
        for start in range(0, len(comps), batch):
            end = start + batch
            print("Processing sources %d - %d" % (start, start + len(fluxes[start:end])))
            predict(mset, mwabeam, ras[start:end], decs[start:end], fluxes[start:end], applybeam=False)

        mset.close()

    if args.uncalibrate or args.noise:
        addImagingColumns(args.mset)
        mset = table(args.mset, readonly=False)
        data = mset.getcol('DATA')

        if args.uncalibrate:
            ant1, ant2 = mset.getcol('ANTENNA1'), mset.getcol('ANTENNA2')
            antids = range(0, mset.ANTENNAS)

            # Leakage
            leakage =  np.random.uniform(-np.pi, np.pi, len(antids))

            # Gains
            gX = np.random.uniform(0.1, 3, len(antids)) * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(antids)))
            gY = np.random.uniform(0.1, 3, len(antids)) * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(antids)))

            # Uncalibration array
            uncalibrators = np.zeros((len(antids), 2, 2), dtype=np.complex)
            uncalibrators[:, 0, 0] = gX * np.cos(leakage)
            uncalibrators[:, 0, 1] = gX * np.sin(leakage)
            uncalibrators[:, 1, 0] = -gY * np.sin(leakage)
            uncalibrators[:, 1, 1] = gY * np.cos(leakage)

            # Uncalibrate data
            uncalibrators_H = np.conj(np.transpose(uncalibrators, [0, 2, 1]))
            for i in range(0, data.shape[1]):
                data[:, i, :] = np.reshape(
                    np.matmul(
                        uncalibrators[ant1],
                        np.matmul(
                            np.reshape(data[:, i, :], (len(ant1), 2, 2)), uncalibrators_H[ant2]
                        )
                    ),
                    (len(ant1), 4)
                )

        if args.noise:
            # Add noise
            data += np.random.normal(0, args.noise, data.shape) + 1j * np.random.normal(0, args.noise, data.shape)

        mset.putcol('CORRECTED_DATA', data)
        mset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--metafits', required=True)
    parser.add_argument('--nosimulate', action='store_true')
    parser.add_argument('--uncalibrate', action='store_true')
    parser.add_argument('--fluxthreshold', type=float, default=0)
    parser.add_argument('--noise', type=float, default=0)
    args = parser.parse_args()
    main(args)

