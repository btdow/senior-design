#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import ceil

from sympy import *
from sympy.polys.groebnertools import *

import numpy as np
from numba import cuda


def cp_cuda(p1, p2, ring):
    """
    Initiates Critical Pair computation on GPU.
    Called in the same way as critical_pair, but
    has many supporting functions.

    All data passed to GPU will be of type int32
    in order to ensure precision and prevent overflow.
    Benchmarks and tests will be run over Galois Fields
    significantly smaller than MAX_INT.

    Prepares data to send to PyCUDA/Numba Cuda.Jit kernel.

    Input: p1, p2 : the labeled polynomials B[i], B[j]
           ring   : just passing through to parser
    """
    mod_pair = modified_pair((p1, p2))
    mod = ring.domain.mod
    nvars = len(ring.symbols)  # array stride
    lt_buf = np.zeros(nvars, dtype=np.uint32)
    fdest = np.zeros(2 * nvars + 1, dtype=np.uint32)
    gdest = np.zeros_like(fdest)

    f = get_cuda_cp_array(mod_pair[0], nvars, mod)
    g = get_cuda_cp_array(mod_pair[1], nvars, mod)

    kernel_data = [nvars, mod, lt_buf, f, g, fdest, gdest]
    cuda_cp_arys = numba_cp_kernel_launch(kernel_data)

    # Modular Inverse function from sympy.
    fdest[-1] = mod_inverse(f[-1], mod)
    gdest[-1] = mod_inverse(g[-1], mod)

    gpu_cp = parse_cuda_cp_to_sympy(cuda_cp_arys, (p1, p2), ring)

    return gpu_cp


def modified_pair(pair):
    """
    Returns truncated lbp pair for transformation
    to cuda arrays.

    Input: pair : a tuple of two labeled polynomials
    Output: modified_pair : a tuple of the components
                            we operate on in critical
                            pair computation Sign(f/g r)
                            multiplier and leading terms
                            of f and g.
    """
    modified_pair = []
    p1 = []
    p2 = []

    # Only get signature multiplier and LT
    p1.append(pair[0][0][0])
    p1.append(Polyn(pair[0]).LT)
    modified_pair.append(tuple(p1))

    p2.append(pair[1][0][0])
    p2.append(Polyn(pair[1]).LT)
    modified_pair.append(tuple(p2))

    return tuple(modified_pair)


def get_cuda_cp_array(mod_pair_part, nvars, mod):
    """
    Fills a np.array for cuda cp with data
    appropriate for CP calculation
    Modifies in place.
    """
    cuda_cp_array = np.zeros(2 * nvars + 1, dtype=np.uint32)

    # Signature Multiplier
    for i, s in enumerate(mod_pair_part[0]):
        cuda_cp_array[i] = s % mod

    # Leading Monomial
    for i, e in enumerate(mod_pair_part[1][0]):
        cuda_cp_array[i + nvars] = e % mod

    # Leading Coefficient
    cuda_cp_array[-1] = mod_pair_part[1][1] % mod

    return cuda_cp_array

def numba_cp_kernel_launch(kernel_data):
    """
    Prepared nparray data for numba cuda jit
    Appears to modify sent arrays in place from
    their documentation
    """
    nvars = kernel_data[0]
    mod = kernel_data[1]
    lt_buf = kernel_data[2]
    f = kernel_data[3]
    g = kernel_data[4]
    fdest = kernel_data[5]
    gdest = kernel_data[6]

    # Launch kernel
    tpb = 32
    bpg = (fdest.size + (tpb - 1)) // tpb

    numba_critical_pairs[tpb, bpg](nvars, mod, lt_buf, f, g, fdest, gdest)

    return [fdest, gdest]


@cuda.jit
def numba_critical_pairs(nvars, mod, lt_buf, f, g, fdest, gdest):
    """
    Numba Cuda.Jit kernel for critical pair computation.

    INPUT:
    nvars: integer, used as array stride

    lt_buf: intermediate storage for monomial_lcm(f, g) (len nvars + 1)
               lt_buf[:nvars] : monomial
               lt_buf[-1] : 1 (ring.domain.one)
        * This and [f:g]dest should be in the shared memory for all threads
          after computation

    f, g : polynomials for cp computation of len 2*nvars + 1
           f[0:nvars] : signature multiplier field
           f[nvars:-1] : leading monomial of f
           f[-1] : leading coefficient of f

    fdest : a destination array for final result
            fdest[:nvars] : sig(fr) multiplier field
            fdest[nvars:2*nvars+1] : um field
            fdest[nvars:2*nvars] : um monomial field
            fdest[2*nvars] : um coefficient
            same for g

    OUTPUT: fdest, gdest arrays appropriately filled.

    Procedure:
    1) Compute lt: max of f[i], g[i] for i in range(nvars, 2*nvars+1)
       (the lt is initialized with 1 as its last entry, so we're good there)
    2) Synchronize Threads
    3) dest, lt should be put into shared memory ?

    4) Compute um and vm simultaneously (no data dependency)
       subtraction for first nvars, division for last entry
       um, vm are stored in their respective fields in dest
    5) Synchronize threads

    6) Compute sig(fr) mult, sig(gr) mult simultaneously (no dependency)
       sum of respective sig in f or g, um or vm fields in dest sig fields.
    7) Synchronize threads
    8) Copy fdest, gdest back to host
    """
    i = cuda.grid(1)
    # lt
    if i < lt_buf.size:
        lt_buf[i] = max(f[nvars + i], g[nvars + i])

    cuda.syncthreads()
    # um vm
    if i < lt_buf.size:
        fdest[nvars + i] = lt_buf[i] - f[nvars + i]
        gdest[nvars + i] = lt_buf[i] - g[nvars + i]

    cuda.syncthreads()
    # sig fr gr
    if i < lt_buf.size:
        fdest[i] = f[i] + fdest[nvars + i]
        gdest[i] = g[i] + gdest[nvars + i]


def parse_cuda_cp_to_sympy(cuda_cp, pair, ring):
    """
    Convert cuda_cp array to sympy's 6-tuple form
    by passing through the parts of pair that are
    unmodified during cp computation

    Input: cuda_cp : a list of 2 numpy arrays with
                     Sign(fr) multiplier in [0:nvars]
                     um [nvars:end] with um coefficient
                     at cuda_cp[-1]

           pair: two labeled polynomials from B
                 indices match cuda_cp arrays

           ring: need it for domain
    """
    nvars = len(ring.symbols)

    gpu_sympy_cp = []

    """
    print("In Parser")
    print("---------")
    print("CUDA Crit Pairs")
    for i, cp in enumerate(cuda_cp):
        print(i, cp)
    print("--------")
    print("Original Pair")
    for i, p in enumerate(pair):
        print(i, p)
    print("-------")
    input("press enter to continue")
    """

    # Build critical pair list
    gpu_sympy_cp.append([cuda_cp[0][:nvars], pair[0][0][1]])  # sig(fr)
    gpu_sympy_cp.append([cuda_cp[0][nvars:-1], cuda_cp[0][-1]])  # um coeff
    gpu_sympy_cp.append(pair[0])  # f
    gpu_sympy_cp.append([cuda_cp[1][:nvars], pair[1][0][1]])  # sig(gr)
    gpu_sympy_cp.append([cuda_cp[1][nvars:-1], cuda_cp[1][-1]]) # vm coeff
    gpu_sympy_cp.append(pair[1])  # g

    # Retuple
    gpu_sympy_cp[0][0] = tuple([s for s in gpu_sympy_cp[0][0]])
    gpu_sympy_cp[1][0] = tuple([e for e in gpu_sympy_cp[1][0]])
    gpu_sympy_cp[3][0] = tuple([s for s in gpu_sympy_cp[3][0]])
    gpu_sympy_cp[4][0] = tuple([e for e in gpu_sympy_cp[4][0]])

    gpu_sympy_cp[0] = tuple(gpu_sympy_cp[0])
    gpu_sympy_cp[1] = tuple(gpu_sympy_cp[1])
    gpu_sympy_cp[3] = tuple(gpu_sympy_cp[3])
    gpu_sympy_cp[4] = tuple(gpu_sympy_cp[4])

    return tuple(gpu_sympy_cp)


if __name__ == "__main__":
    print("GPU Critical Pairs Test")

    print("Cyclic Affine 4")
    r, a, b, c, d = ring(symbols="a, b, c, d", domain=GF(65521), order="grevlex")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    F = [f1, f2, f3, f4]

    """
    print("Katsura Affine 4")
    r, x1, x2, x3, x4 = ring(symbols='x1, x2, x3, x4', domain=GF(65521), order='grevlex')
    f1 = x1 + 2*x2 + 2*x3 + 2*x4 - 1
    f2 = x1**2 + 2*x2**2 + 2*x3**2 + 2*x4**2 - x1
    f3 = 2*x1*x2 + 2*x2*x3 + 2*x3*x4 - x2
    f4 = x2**2 + 2*x1*x3 + 2*x2*x4 - x3
    F = [f1, f2, f3, f4]
    """

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]
    CP = [critical_pair(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    GPU_CP = [cp_cuda(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    
    for gcp, cp in zip(GPU_CP, CP):
        print("GPU Critical Pair: ")
        for i, part in enumerate(gcp):
            print(i, part)
        print('-------------------')
        print("Original Critical Pair {}:".format(i))
        for i, part in enumerate(cp):
            print(i, part)
        print('-------------------')

    assert(CP == GPU_CP)
