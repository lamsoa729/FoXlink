#!/usr/bin/env python

"""@package docstring
File: create_me_1D_avg_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from numba import njit
from scipy.integrate import quad
from .rod_motion_solver import get_rod_drag_coeff
from .me_zrl_odes import (dmu00_dt_zrl_1D,
                          dmu10_dt_zrl_1D)
from .me_zrl_helpers import (semi_anti_deriv_boltz_0, semi_anti_deriv_boltz_1)


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


def get_src_term_arr(x, ui, L, co, ks, beta):
    """TODO: Docstring for get_src_term_arr.

    @param arg1 TODO
    @return: TODO

    """
    q00 = co * fast_1D_src_kl(L, x, ui, 1., ks, beta, 0, 0)
    q10_p = co * fast_1D_src_kl(L, -x, 1., ui, ks, beta, 0, 1)
    q01_p = co * fast_1D_src_kl(L, x, ui, 1., ks, beta, 0, 1)
    q10_n = co * fast_1D_src_kl(L, -x, -1., ui, ks, beta, 0, 1)
    q01_n = co * fast_1D_src_kl(L, x, ui, -1., ks, beta, 0, 1)
    return [q00, q10_p, q01_p, q00, q10_n, q01_n]


def fast_1D_src_kl(L, x, ui, uj, ks, beta, k=0, l=0):
    """!TODO: Docstring for fast_zrl_src_kl

    @return: TODO

    """
    if l == 0:
        integrand = fast_1D_src_integrand_l0
    elif l == 1:
        integrand = fast_1D_src_integrand_l1
    else:
        raise RuntimeError(
            "{}-order derivatives have not been implemented for fast source solver.".format(l))
    sigma = np.sqrt(2. / (ks * beta))
    q, _ = quad(integrand, -.5 * L, .5 * L,
                args=(L, x, ui, uj, sigma, k))
    return q


@njit
def fast_1D_src_integrand_l0(s_i, L, x, ui, uj, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.
    @return: TODO

    """
    A = -1. * uj * (x + (ui * s_i))
    exponent = -1. * (s_i * (s_i + 2. * ui * uj) -
                      (A * A)) / (sigma * sigma)

    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_0(-.5 * L, sigma, A)
    I_p = semi_anti_deriv_boltz_0(.5 * L, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_1D_src_integrand_l1(s_i, L, x, ui, uj, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k1.
    @return: TODO

    """
    A = -1. * uj * (x + (ui * s_i))
    exponent = -1. * (s_i * (s_i + 2. * ui * uj) -
                      (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_1(-.5 * L, sigma, A)
    I_p = semi_anti_deriv_boltz_1(.5 * L, sigma, A)
    return pre_fact * (I_p - I_m)


def me_1D_avg_evolver(sol, gpara, params):
    """TODO: Docstring for me_1D_avg_evolver.

    @param sol TODO
    @param gpara TODO
    @param pdict TODO
    @return: TODO

    """
    # Get system parameters
    beta = params['beta']
    N = params['rod_dense']
    P = params['polarity']
    L = params['length']
    # Get motor parameters
    co = params['co']
    vo = params['vo']
    ks = params['ks']
    fs = params['fs']
    ko = params['ko']
    kappa = vo * ks / fs
    # Get source terms
    q_arr = []
    q_arr += get_src_term_arr(sol[0], 1., L, co, beta)
    q_arr += get_src_term_arr(sol[1], -1., L, co, beta)
    q_arr += get_src_term_arr(sol[2], 1., L, co, beta)
    q_arr += get_src_term_arr(sol[3], -1., L, co, beta)
    # Evolve positions
    dx_arr = [dx_dt_avg(sol[0], 1., sol[4:10], N, P, gpara, ks),
              dx_dt_avg(sol[1], -1., sol[10:16], N, P, gpara, ks),
              dx_dt_avg(sol[2], 1., sol[16:22], N, P, gpara, ks),
              dx_dt_avg(sol[3], -1., sol[22:], N, P, gpara, ks)]
    # Evolve moments
    dmu_arr = calc_1D_moment_derivs(sol[0], 1., 1., sol[4:7], q_arr[:3],
                                    ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[0], 1., -1., sol[7:10], q_arr[3:6],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[1], 1., 1., sol[10:13], q_arr[6:9],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[1], 1., -1., sol[13:16], q_arr[9:12],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[2], 1., 1., sol[16:19], q_arr[12:15],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[2], 1., -1., sol[19:22], q_arr[15:18],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[3], 1., 1., sol[22:25], q_arr[18:21],
                                     ko, vo, kappa)
    dmu_arr += calc_1D_moment_derivs(sol[3], 1., -1., sol[25:], q_arr[21:],
                                     ko, vo, kappa)
    dsol = np.concatenate(dx_arr, dmu_arr)
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError('Infinity or NaN thrown in ODE solver derivatives. '
                           'Current derivatives', dsol)
    return dsol


@njit
def calc_1D_moment_derivs(x, u_i, u_j, mu_kl, q_arr, ko, vo, kappa):
    dmu00 = dmu00_dt_zrl_1D(mu_kl[0], ko, q_arr[0])
    dmu10 = dmu10_dt_zrl_1D(mu_kl[0], mu_kl[1], mu_kl[2], x * u_i, u_i * u_j,
                            ko, vo, kappa, q_arr[1])
    dmu01 = dmu10_dt_zrl_1D(mu_kl[0], mu_kl[2], mu_kl[1], -x * u_j, u_i * u_j,
                            ko, vo, kappa, q_arr[2])
    return [dmu00, dmu10, dmu01]


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
