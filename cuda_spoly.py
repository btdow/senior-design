#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import ceil
from itertools import chain

from sympy import *
from sympy.polys.groebnertools import *
from sympy.polys.orderings import monomial_key

import numpy as np
from numba import cuda
from cuda_cp import cp_cuda

def cuda_s_poly(cp, B, r):
    """
    Execute as a script to test.
    Called slightly differently from s_poly,
    must include ring.

    Prepare the data for the s-polynomial

    Create numpy arrays to send to gpu
    f, g, and dest arrays must all be the same length,

    figuring out the exact required output dimensions
    of the spoly procedure is exactly the F4 symbolic
    preprocessing step, and I don't know of another
    way to do it. Only the subtraction step of spoly
    is carried out on the GPU because of this, but
    it provides a micro demonstration of an F4 style
    matrix reduction. 


    Input: cp : a critical pair
           ring: for ordering, modulus
    """
    # Left and right of critical pair
    Ld = [(cp[0], cp[1], cp[2]), (cp[3], cp[4], cp[5])]

    # Get Length/monomials of destination array
    spair_info = symbolic_preprocessing(Ld, B, r)

    gpu_spoly = spoly_numba_io(spair_info, r)

    return gpu_spoly


def cuda_s_poly2(cp, r):
    """
    Another version of s_poly that
    just calculates each step in separate kernels.
    and reindexes the monomials on the host
    in between. May be improved by use of
    a cuda stream in CUDA-C or PyCUDA
    """
    nvars = len(r.symbols)
    mod = r.domain.mod
    # Multiply step
    f = cp[2]
    g = cp[5]
    
    fsig_idx = Sign(f)[1]
    gsig_idx = Sign(g)[1]
    fnum = Num(f)
    gnum = Num(g)

    um = cp[1][0]
    vm = cp[4][0]
    uc = cp[1][1]
    vc = cp[4][1]
    um = np.array(um, dtype=np.uint32)
    vm = np.array(vm, dtype=np.uint32)

    uv_coeffs = [uc, vc]
    uv_coeffs = np.array(uv_coeffs, dtype=np.int64)

    fsm = [Sign(f)[0]] + Polyn(f).monoms()
    fsm = np.array(fsm, dtype=np.uint32)

    gsm = [Sign(g)[0]] + Polyn(g).monoms()
    gsm = np.array(gsm, dtype=np.uint32)

    fc = [f for f in Polyn(f).coeffs()]
    fc = np.array(fc, dtype=np.int64)
    gc = [g for g in Polyn(g).coeffs()]
    gc = np.array(gc, dtype=np.int64)

    fsm_dest = np.zeros_like(fsm)
    gsm_dest = np.zeros_like(gsm)
    fc_dest = np.zeros_like(fc)
    gc_dest = np.zeros_like(gc)

    total_monoms_sigs = len(Polyn(f).terms()) + len(Polyn(g).terms()) + 2
    
    # prepare threads
    threadsperblock = (32, 32)
    blockspergrid_x = (total_monoms_sigs + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (um.size + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # launch kernel
    spoly_mul_numba_kernel[blockspergrid,
                           threadsperblock](fsm_dest, gsm_dest, fc_dest,
                                            gc_dest, fsm, gsm, fc, gc, um, vm,
                                            uv_coeffs)

    # Get all monomials in both umf, vmg, sort by ordering, reindex
    # f, g in a 2d coefficient array, send to other kernel
    if sum(sum(fsm_dest)) == 0:
        if sum(sum(gsm_dest)) == 0:
            return (((r.zero_monom), 0),
                    r.from_expr('0'), 0)

    fnew = [tuple(f) for f in fsm_dest]
    gnew = [tuple(g) for g in gsm_dest]
    fnew_sig = fnew[0]
    gnew_sig = gnew[0]
    fnew_monoms = [f for f in fnew[1:]]
    gnew_monoms = [g for g in gnew[1:]]

    all_monoms = set(fnew_monoms).union(set(gnew_monoms))
    all_monoms = sorted(all_monoms, key=monomial_key(order=r.order), reverse=True)

    spair_matrix = np.zeros((2, len(all_monoms)), dtype=np.uint32)

    for fm, fc in zip(fnew_monoms, fc_dest):
        spair_matrix[0, all_monoms.index(fm)] = fc
    for gm, gc in zip(gnew_monoms, gc_dest):
        spair_matrix[1, all_monoms.index(gm)] = gc

    # Parse
    spair_info = parse_gpu_spoly_mul(spair_matrix, all_monoms, fnew_sig,
                                     gnew_sig, fsig_idx, gsig_idx, fnum,
                                     gnum, r)

    if not spair_info:
        return (((r.zero_monom), 0), r.from_expr('0'), 0)

    lb_spoly = spoly_numba_io(spair_info, r)

    return lb_spoly


def spoly_numba_io(spair_info, r):
    """
    Prepare the mini macaulay matrix for the numba kernel
    Called after symbolic_preprocessing only.

    Coefficient values should be of size int32

    All vals will be signed int32.
    """
    cols = spair_info["cols"]
    rows = spair_info["rows"]

    spair_matrix = np.zeros((rows, cols), dtype=np.int32)
    dest = np.zeros(cols, dtype=np.int32)

    # fill at coordinates with nonzero entries
    for coords in spair_info["nze"]:
        spair_matrix[coords[0][0], coords[0][1]] = coords[1]
    
    threadsperblock = 32
    blockspergrid = (dest.size + (threadsperblock - 1)) // threadsperblock

    spoly_sub_numba_kernel[threadsperblock, blockspergrid](dest, spair_matrix)

    # parse
    lb_spoly = parse_gpu_spoly(dest, spair_info, r)

    return lb_spoly


@cuda.jit
def spoly_sub_numba_kernel(dest, spair):
    """
    Basically Micro F4 partial reduction

    Subtracts f from g and stores in dest
    spair is a 2-row macaulay matrix of 
    coefficients in f and g in given monomial ordering.
    
    Likely grossly inefficient compared to CPU due
    to memory access times, but parallel. Demonstrates
    part of the process of F4 reduction.
    """
    pos = cuda.grid(1)
    if pos < dest.size:
        dest[pos] = spair[0][pos] - spair[1][pos]


@cuda.jit
def spoly_mul_numba_kernel(fsm_dest, gsm_dest, fc_dest, gc_dest,
                           fsm, gsm, fc, gc, um, vm, uv_coeffs):
    """
    Numba lbp_mul kernel for cuda_s_poly2. 
    Stage one of Spoly, 
    fsm_dest, gsm_dest must be made a set, sorted, 
    and fc, gc reindexed into a 2d array for 
    sub step kernel
    """
    i, j = cuda.grid(2)
    if j < fsm_dest.shape[0]:
        if i < um.size:
            fsm_dest[j, i] = um[i] + fsm[j, i]

    if j < gsm_dest.shape[0]:
        if i < vm.size:
            gsm_dest[j, i] = vm[i] + gsm[j, i]

    if i < fc_dest.size:
        fc_dest[i] = (uv_coeffs[0] * fc[i]) % 65521

    if j < gc_dest.size:
        gc_dest[i] = (uv_coeffs[1] * gc[i]) % 65521


def symbolic_preprocessing(Ld, B, r):
    """
    Mini Symbolic Preprocessing for Single S-Polynomial
    
    Input: Ld     : two 3-tuples(sig, um, f), (sig, vm, g)
           B      : intermediate basis
           ring   : for domain, order stuff
    
    Out: Information needed to construct a macaulay matrix.
    """
    domain = r.domain

    Fi = [lbp_mul_term(sc[2], sc[1]) for sc in Ld]
    Done = set([Polyn(f).LM for f in Fi])
    M = [Polyn(f).monoms() for f in Fi]
    M = set([i for i in chain(*M)]).difference(Done)
    while M != Done:
        MF = M.difference(Done)
        if MF != set():
            m = MF.pop()
            Done.add(m)
            for g in B:
                if monomial_divides(Polyn(g).LM, m):
                    u = term_div((m, domain.one), Polyn(g).LT, domain)
                    ug = (lbp_mul_term(g, u))
                    for m in Polyn(ug).monoms():
                        M.add(m)
        else:
            break

    Done = sorted(Done, key=monomial_key(order=r.order), reverse=True)

    # pseudo COO sparse format
    nonzero_entries = []
    for i, f in enumerate(Fi):
        for t in Polyn(f).terms():
            nonzero_entries.append(((i, Done.index(t[0])), t[1]))

    spair_info = dict()
    spair_info["cols"] = len(Done)
    spair_info["rows"] = len(Fi)
    spair_info["nze"] = nonzero_entries
    spair_info["monomials"] = Done
    spair_info["spair"] = Fi

    return spair_info


def parse_gpu_spoly(dest, spair_info, r):
    """
    Return GPU spoly to sympy labeled polynomial

    Input: dest : the destination array from kernel
           spair_info: from symbolic_preprocessing
           ring : ordering, domain, etc.

    Output: sympy lbp 3 tuple (sig, poly, num)
    """
    spoly_sig = None
    spoly_num = None

    f = spair_info["spair"][0]
    g = spair_info["spair"][1]
    
    if lbp_cmp(f, g) == -1:
        spoly_sig = Sign(g)
        spoly_num = Num(g)
    elif lbp_cmp(f, g) == 1: 
        spoly_sig = Sign(f)
        spoly_num = Num(f)

    if spair_info["monomials"] == [r.zero_monom]:
        return (spoly_sig, Polyn(f), spoly_num)

    pexp = []
    for i, c in enumerate(dest):
        if c != 0:
            pexp.append('+' + str(c))
            for j, e in enumerate(spair_info["monomials"][i]):
                if e != 0:
                    pexp.append('*' + str(r.symbols[j]) + '**' + str(e))
    if pexp != []:
        spol = r.from_expr(''.join(pexp))
        lb_spol = tuple([spoly_sig, spol, spoly_num])
        return lb_spol
    return (((r.zero_monom), 0), r.from_expr('0'), 0)


def parse_gpu_spoly_mul(spair_matrix, all_monoms, fnew_sig, gnew_sig,
                        fsig_idx, gsig_idx, fnum, gnum, r):
    """
    parse into same output as symbolic_preprocessing to reuse
    numba_spoly_io function. Contains some redundant information,
    optimize later.
    """
    fsig = (tuple(fnew_sig), fsig_idx)
    gsig = (tuple(gnew_sig), gsig_idx)

    if all_monoms == [r.zero_monom] or sum(sum(spair_matrix)) == 0 or all_monoms == []:
        return None

    fpexp = []
    gpexp = []

    for i, c in enumerate(spair_matrix[0]):
        if c != 0:
            fpexp.append('+' + str(c))
            for j, e in enumerate(all_monoms[i]):
                if e != 0:
                    fpexp.append('*' + str(r.symbols[j]) + '**' + str(e))
    fp = r.from_expr(''.join(fpexp))
    lbf = (fsig, fp, fnum)

    for i, c in enumerate(spair_matrix[1]):
        if c != 0:
            gpexp.append('+' + str(c))
            for j, e in enumerate(all_monoms[i]):
                if e != 0:
                    gpexp.append('*' + str(r.symbols[j]) + '**' + str(e))

    gp = r.from_expr(''.join(gpexp))
    lbg = (gsig, gp, gnum)

    spair = [lbf, lbg]

    nze = []
    for i, f in enumerate(spair):
        for t in Polyn(f).terms():
            nze.append(((i, all_monoms.index(t[0])), t[1]))

    spair_info = dict()
    spair_info["rows"] = 2
    spair_info["cols"] = len(all_monoms)
    spair_info["spair"] = spair
    spair_info["monomials"] = all_monoms
    spair_info["nze"] = nze

    return spair_info


if __name__ == "__main__":
    print("CUDA Spoly Test")

    """
    r, a, b, c, d, e = ring(symbols='a, b, c, d, e', domain=GF(65521), order='grevlex')
    print("Cyclic Affine 4")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1

    print("Cyclic Homogeneous 4")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*c + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - e**4
    F = [f1, f2, f3, f4]
    """

    print("Katsura Affine 4")
    r, x1, x2, x3, x4 = ring(symbols="x1, x2, x3, x4", domain=GF(65521), order='grevlex')
    f1 = x1 + 2*x2 + 2*x3 + 2*x4 - 1
    f2 = x1**2 + 2*x2**2 + 2*x3**2 + 2*x4**2 - x1
    f3 = 2*x1*x2 + 2*x2*x3 + 2*x3*x4 - x2
    f4 = x2**2 + 2*x1*x3 + 2*x2*x4 - x3
    F = [f1, f2, f3, f4]

    order = r.order

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]
    B = sorted(B, key=lambda g: order(Polyn(g).LM), reverse=True)

    CP = [cp_cuda(B[i], B[j], r)
           for i in range(len(B)) for j in range(i + 1, len(B))]
    CP = sorted(CP, key=lambda cp: cp_key(cp, r), reverse=True)

    S = [cuda_s_poly(CP[i], B, r) for i in range(len(CP))]
    S2 = [cuda_s_poly2(CP[i], r) for i in range(len(CP))]
    S_orig = [s_poly(CP[i]) for i in range(len(CP))]

    print("---------------------")
    for i in range(len(S)):
        print("Original S-Poly: {}".format(i))
        print(S_orig[i])
        print("---------------------")
        print("GPU S-Poly v1: {}".format(i))
        print(S[i])
        print("Equal? ", S[i] == S_orig[i])
        print("---------------------")
        print("GPU S-Poly v2: {}".format(i))
        print(S2[i])
        print("Equal? ", S2[i] == S_orig[i])
        print("Equal to S? ", S[i] == S2[i])
        print("---------------------")
