#!/usr/bin/env python

"""@package docstring
File: create_me_1D_avg_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from numba import njit
from .rod_motion_solver import get_rod_drag_coeff
from .me_zrl_odes import (dmu00_dt_zrl_1D,
                          dmu10_dt_zrl_1D)


@njit
def dx_dt_avg(x, u, mukl, N, P, gpara, ks):
    """TODO: Docstring for dx_dt.

    @param x TODO
    @param u TODO
    @param mukl array of mukls up to first order 0 mu00P, 1 mu10P, ..., 5 mu01N
    @param gpara TODO
    @return: TODO

    """
    a = .5 * ks * N / gpara
    return a * ((1. + P) * (x * mukl[0] + u * mukl[1] - mukl[2])
                + (1. - P) * (x * mukl[3] + u * mukl[4] + mukl[5]))


def me_1D_avg_evolver(sol, gpara, params):
    """TODO: Docstring for me_1D_avg_evolver.

    @param sol TODO
    @param gpara TODO
    @param pdict TODO
    @return: TODO

    """
    beta = params['beta']
    ks = params['ks']
    N = params['rod_dense']
    P = params['polarity']
    # Get source terms
    # Evolve positions
    sol[0] = dx_dt_avg(sol[0], 1., sol[4:10], N, P, gpara, ks)
    sol[1] = dx_dt_avg(sol[1], -1., sol[10:16], N, P, gpara, ks)
    sol[2] = dx_dt_avg(sol[2], 1., sol[16:22], N, P, gpara, ks)
    sol[3] = dx_dt_avg(sol[3], -1., sol[22:], N, P, gpara, ks)
    # Evolve moments
    pass


def calc_1D_moment_derivs(x, u, mu_kl, N, P, q_arr):

    pass


def init_me_1D_avg_evolver(slvr, sol_init):
    """!Create a closure for ode solver

    @param slvr: MomentExpansion1DAvgSolver solver class
    @param sol_init: Array of time-dependent variables in the ODE
    @return: evolver function for ODE of interest

    """
    gpara, _, _ = get_rod_drag_coeff(slvr.visc, slvr.L, slvr.rod_diam)

    def me_1D_avg_evolver_closure(t, sol):
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(
                ('Infinity or NaN thrown in ODE solver solutions.'
                 ' Current solution: '), sol)
        return me_1D_avg_evolver(sol, gpara, slvr.__dict__)
    return me_1D_avg_evolver_closure


##########################################
