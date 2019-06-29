from __future__ import division, print_function

import math
import time as tm
import threading

from astropy.time import Time
import astropy.units as units
from casacore.tables import taql
from numba import njit, cuda, float32, complex64, void, prange
import numpy as np

from astrobits.coordinates import radec_to_lm


def predict(mset, mwabeam, ras, decs, fluxes, applybeam=True):
    ra0, dec0 = mset.FIELD.getcell('PHASE_DIR', 0)[0]
    freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
    lambdas = 299792458 / freqs
    unique_times = sorted(set(mset.getcol('TIME')))

    # Calculate l, m, n based on current phase center
    ls, ms = radec_to_lm(ras, decs, ra0, dec0)
    ns = np.sqrt(1 - ls**2 - ms**2)
    ls, ms, ns = float32(ls), float32(ms), float32(ns)

    for i, unique_time in enumerate(unique_times):
        print("Time interval: %d/%d" % (i+1, len(unique_times)))
        tbl = taql("select UVW, DATA from $mset where TIME = $unique_time and not FLAG_ROW and ANTENNA1 <> ANTENNA2")
        uvw, data = tbl.getcol('UVW'), tbl.getcol('DATA')

        u_lambda, v_lambda, w_lambda = np.float32(uvw.T[:, :, None] / lambdas)
        u_lambda, v_lambda, w_lambda = u_lambda.copy(), v_lambda.copy(), w_lambda.copy()

        julian_date = unique_time // (24*60*60) + (unique_time % (24*60*60) / (24*60*60))
        julian_date = Time(julian_date, format='mjd')
        mwabeam.time = julian_date

        # Prepare I_app
        start = tm.time()
        I = np.zeros((len(fluxes), len(freqs), 2, 2))
        jones = np.zeros((len(fluxes), len(freqs), 2, 2), dtype=np.complex128)
        I[:, :, 0, 0] = fluxes  # XX
        I[:, :, 1, 1] = fluxes  # YY

        if applybeam:
            chunksize = 64
            assert(len(freqs) % chunksize == 0)
            midfreqs = np.mean(np.reshape(freqs, (-1, 64)), axis=1)
            idx = np.repeat(range(len(freqs) // chunksize), chunksize)
            jones = mwabeam.jones(ras, decs, midfreqs)[:, idx]
            jones_H = np.conj(np.transpose(jones, axes=[0, 1, 3, 2]))
            I_app = np.matmul(jones, np.matmul(I, jones_H))
            I_app = np.reshape(I_app, (len(fluxes), len(freqs), 4))
            I_app = np.complex64(I_app)
        else:
            I_app = np.complex64(np.reshape(I, (len(fluxes), len(freqs), 4)))
        print("Beam calculation elapsed: %g" % (tm.time() - start))

        # Predict
        start = tm.time()
        tpb = (25, 25)
        bpg = (data.shape[0] // 25 + 1, data.shape[1] // 25 + 1)

        def _thread(data, ngpu, start, end):
            with cuda.gpus[ngpu]:
                cudapredict[bpg, tpb](
                    data,
                    u_lambda,
                    v_lambda,
                    w_lambda,
                    I_app[start:end],
                    ls[start:end],
                    ms[start:end],
                    ns[start:end],
                )

        # Send to multiple GPUs if present
        # ngpus = len(cuda.gpus)
        # batch = len(fluxes) // ngpus + 1
        # threads = []
        # datas = []
        # for i in range(0, ngpus):
        #     print("Starting work on gpu %d..." % i)
        #     datas.append(data.copy())
        #     thread = threading.Thread(target=_thread, args=(
        #         datas[i], i, batch * i, batch * (i + 1)
        #     ))
        #     threads.append(thread)
        #     thread.start()
        #     print("Started")

        # # Wait for all GPUs to complete
        # for i, thread in enumerate(threads):
        #     print("Waiting for device %d to finish..." % i)
        #     thread.join()
        #     print("Done")
        # cpupredict(data, u_lambda, v_lambda, w_lambda, I_app, ls, ms, ns)
        cudapredict[bpg, tpb](data, u_lambda, v_lambda, w_lambda, I_app, ls, ms, ns)

        print("Prediction elapsed: %g" % (tm.time() - start))
        tbl.putcol('DATA', data)
        tbl.flush()


@njit([
    void(complex64[:, :, :], float32[:, :], float32[:, :], float32[:, :], complex64[:, :, :], float32[:], float32[:], float32[:])
], parallel=True)
def cpupredict(data, u_lambda, v_lambda, w_lambda, I_app, ls, ms, ns):
    # Parallelize over points
    for i in prange(0, len(I_app)):
        for j in range(0, u_lambda.shape[0]):
            for k in range(0, u_lambda.shape[1]):
                point = np.exp(2j * np.pi * (
                    u_lambda[j, k] * ls[i] + v_lambda[j, k] * ms[i] + w_lambda[j, k] * (ns[i] -1)
                ))
                data[j, k] += I_app[i, k, :] * point


@cuda.jit(
    void(complex64[:, :, :], float32[:, :], float32[:, :], float32[:, :], complex64[:, :, :], float32[:], float32[:], float32[:]),
    fastmath=True,
)
def cudapredict(data, u_lambda, v_lambda, w_lambda, I_app, ls, ms, ns):
    nrow, nchan = cuda.grid(2)

    if nrow >= data.shape[0] or nchan >= data.shape[1]:
        return

    tmp = cuda.local.array(4, dtype=complex64)
    tmp[:] = 0
    u, v, w = u_lambda[nrow, nchan], v_lambda[nrow, nchan], w_lambda[nrow, nchan]

    for npoint in range(0, I_app.shape[0]):
        phase = 2 * math.pi * (
            u * ls[npoint] +
            v * ms[npoint] +
            w * (ns[npoint] - 1)
        )
        expphase = math.cos(phase) + 1j * math.sin(phase)

        for pol in range(0, 4):
            tmp[pol] += I_app[npoint, nchan, pol] * expphase

    for pol in range(0, 4):
        data[nrow, nchan, pol] += tmp[pol]
