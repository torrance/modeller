#! /usr/bin/env python
from __future__ import print_function, division

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
from casacore.tables import addImagingColumns, taql
import numpy as np

from radical.coordinates import radec_to_lm
from radical.measurementset import MeasurementSet
from radical.mwabeam import MWABeam
import radical.skymodel as skymodel


mwabeam = MWABeam('modelled.metafits')

with open('model.txt') as f:
    models = np.array(skymodel.parse(f))

comps = np.array([comp for model in models for comp in model.components])
ras = np.array([comp.ra for comp in comps])
decs = np.array([comp.dec for comp in comps])

print("Simulating %d components" % len(comps))

# Set beam for each component
for comp in comps:
    comp.beam = mwabeam

if True:
    addImagingColumns('modelled.ms')

    mset = MeasurementSet('modelled.ms', refant=8, datacolumn='DATA')
    unique_times = sorted(set(mset.getcol('TIME')))

    ls, ms = radec_to_lm(ras, decs, mset.ra0, mset.dec0)
    ns = np.sqrt(1 - ls**2 - ms**2)

    for k, unique_time in enumerate(unique_times):
        print("Time interval: %d/%d" % (k+1, len(unique_times)))
        _mset = mset.mset
        tbl = taql("select UVW, DATA from $_mset where TIME = $unique_time and not FLAG_ROW and ANTENNA1 <> ANTENNA2")
        uvw, data = tbl.getcol('UVW'), tbl.getcol('DATA')

        u_lambda, v_lambda, w_lambda = uvw.T[:, :, None] / mset.lambdas
        data[:] = 0

        julian_date = unique_time // (24*60*60) + (unique_time % (24*60*60) / (24*60*60))
        julian_date = Time(julian_date, format='mjd')
        mwabeam.time = julian_date

        for j, freq in enumerate(mset.freqs):
            fluxes = np.array([comp.flux(freq) for comp in comps])

            I = np.zeros((len(comps), 2, 2))  # [XX, XY, YX, YY]
            I[:, 0, 0] = fluxes  # XX
            I[:, 1, 1] = fluxes  # YY

            jones = mwabeam.jones(ras, decs, freq)
            jones_H = np.conj(np.transpose(jones, axes=[0, 2, 1]))
            I_app = np.matmul(jones, np.matmul(I, jones_H))
            I_app = np.reshape(I_app, (len(comps), 4))

            if j ==0:
                idx = np.argmax(I_app[:, 0])
                print(I_app[idx])

            points = np.exp(2j * np.pi * (u_lambda[:, j][None, :] * ls[:, None] + v_lambda[:, j][None, :] * ms[:, None] + w_lambda[:, j][None, :] * (ns[:, None] - 1)))
            # point = [ comps, uvw ]

            data[:, j, :] = np.sum(I_app[:, None, :] * points[:, :, None], axis=0)

        tbl.putcol('DATA', data)

    mset.close()

mset = MeasurementSet('modelled.ms', refant=8, datacolumn='DATA')
data = mset.data
ant1, ant2 = mset.ant1, mset.ant2

# Leakage
leakage =  np.random.uniform(-np.pi, np.pi, len(mset.antids))

# Gains
gX = np.random.uniform(0.1, 3, len(mset.antids)) * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(mset.antids)))
gY = np.random.uniform(0.1, 3, len(mset.antids)) * np.exp(1j * np.random.uniform(-np.pi, np.pi, len(mset.antids)))

# Uncalibration array
uncalibrators = np.zeros((len(mset.antids), 2, 2), dtype=np.complex)
uncalibrators[:, 0, 0] = gX * np.cos(leakage)
uncalibrators[:, 0, 1] = gX * np.sin(leakage)
uncalibrators[:, 1, 0] = -gY * np.sin(leakage)
uncalibrators[:, 1, 1] = gY * np.cos(leakage)

# Uncalibrate data
uncalibrators_H = np.conj(np.transpose(uncalibrators, [0, 2, 1]))
for i, freqs in enumerate(mset.freqs):
    data[:, i, :] = np.reshape(
        np.matmul(
            uncalibrators[ant1],
            np.matmul(
                np.reshape(data[:, i, :], (len(ant1), 2, 2)), uncalibrators_H[ant2]
            )
        ),
        (len(ant1), 4)
    )

# Add noise
data += np.random.normal(0, 45, data.shape)

mset.filtered.putcol('CORRECTED_DATA', data)
mset.close()


# Todo
#
# * add bandpass
# * add directional polarisation leakage factor


