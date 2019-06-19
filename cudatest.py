#! /usr/bin/env python

import math

from numba import cuda, float32
import numpy as np



@cuda.jit
def adder(a, b):
  tx = cuda.threadIdx.x
  ty = cuda.blockIdx.x
  bw = cuda.blockDim.x

  pos = tx + ty * bw
  if pos < a.size:
      a[pos] += b[pos]


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)

    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


# A = np.ones((512, 512))
# B = np.ones((512, 512))
# C = np.zeros((512, 512))

# matmul[(128, 128), (4, 4)](A, B, C)
# print(C)


data = np.zeros((1000, 768, 4), dtype=np.complex64)
freqs = np.linspace(130e6, 170e6, 768, dtype=np.float32)
lambdas = np.float32(299792458) / freqs
u = np.linspace(-3000, 3000, 1000, dtype=np.float32)[:, None] / lambdas
v = u.copy()
w = np.zeros_like(u)
A = np.float32(np.random.uniform(1, 10, (1000, 768)))
ls = np.float32(np.random.uniform(-0.5, 0.5, 1000))
ms = np.float32(np.random.uniform(-0.5, 0.5, 1000))
ns = np.float32(np.sqrt(1 - ls**2 - ms**2))

@cuda.jit
def predict(data, u_lambda, v_lambda, w_lambda, A, ls, ms, ns):
    nrow, nchan, npoint = cuda.grid(3)

    if nrow < data.shape[0] and nchan < data.shape[1] and npoint < A.shape[0]:
        for i in range(0, 4):
            #data[nrow, nchan, i] += A[npoint, nchan]
            phase = 2 * math.pi * (
                u_lambda[nrow, nchan] * ls[npoint] +
                v_lambda[nrow, nchan] * ms[npoint] +
                w_lambda[nrow, nchan] * ns[npoint]
            )
            data[nrow, nchan, i] += A[npoint, nchan] * (math.cos(phase) + 1j * math.sin(phase))


predict[(1000, 768, 20), (1, 1, 50)](data, u, v, w, A, ls, ms, ns)
print(data)
