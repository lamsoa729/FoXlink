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

# TODO Move to a separate file ################################################


@njit
def boltz_fact_1D(si, sj, x, ui, uj, ks, beta):
    """Boltzmann factor for 1D scenarios

    @param si TODO
    @param sj TODO
    @param x TODO
    @param ui TODO
    @param uj TODO
    @param ks TODO
    @param beta TODO
    @return: TODO

    """
    return np.exp(-.5 * ks * beta *
                  (x**2 + si**2 + sj**2 + 2. * x *
                   (si * ui - sj * uj) - 2. * ui * uj * si * sj))


@njit
def weighted_boltz_fact_1D(si, sj, x, ui, uj, ks, beta, k=0, l=0):
    """Boltzmann factor for 1D scenarios weightbed by powers of si and sj
    """
    return (np.power(si, k) * np.power(sj, l) *
            boltz_fact_1D(si, sj, x, ui, uj, ks, beta))


# @njit
def dx_dt(x, ui, uj, mukl, gpara, ks):
    """TODO: Docstring for dx_dt.

    @param x TODO
    @param u TODO
    @param mukl array of mukls up to first order 0 mu00P, 1 mu10P, ..., 5 mu01N
    @param gpara TODO
    @return: TODO

    """
    print('x: {}, ui: {}, uj: {}, mukl: [{}, {}, {}], gpara: {}, ks: {}'.format(
        x, ui, uj, mukl[0], mukl[1], mukl[2], gpara, ks))

    a = -1. * (ks / gpara)
    result = a * ((x * mukl[0]) + (ui * mukl[1]) - (uj * mukl[2]))
    # result = a * ((ui * mukl[1]) - (uj * mukl[2]))
    # print('result:', result)
    return result


def calc_1D_moment_derivs(x, u_i, u_j, mu_kl, q_arr, ko, vo, kappa):
    dmu00 = dmu00_dt_zrl_1D(mu_kl[0], ko, q_arr[0])
    dmu10 = dmu10_dt_zrl_1D(mu_kl[0], mu_kl[1], mu_kl[2], x * u_i, u_i * u_j,
                            ko, vo, kappa, q_arr[1])
    dmu01 = dmu10_dt_zrl_1D(mu_kl[0], mu_kl[2], mu_kl[1], -x * u_j, u_i * u_j,
                            ko, vo, kappa, q_arr[2])
    return [dmu00, dmu10, dmu01]


def get_src_term_arr(x, ui, uj, L, co, ks, beta):
    """TODO: Docstring for get_src_term_arr.

    @param arg1 TODO
    @return: TODO

    """
    q00 = co * fast_1D_src_kl(L, x, ui, uj, ks, beta, 0, 0)
    q10 = co * fast_1D_src_kl(L, -x, uj, ui, ks, beta, 0, 1)
    # q10 = co * fast_1D_src_kl(L, x, ui, uj, ks, beta, 1, 0)
    q01 = co * fast_1D_src_kl(L, x, ui, uj, ks, beta, 0, 1)
    return [q00, q10, q01]


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
    A = -1. * (uj * x + uj * ui * s_i)
    exponent = -1. * (x**2 + s_i**2 + 2. * ui * x *
                      s_i - A**2) / (sigma**2)
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
    exponent = -1. * (x * x + s_i * (s_i + 2. * ui * x) -
                      (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_1(-.5 * L, sigma, A)
    I_p = semi_anti_deriv_boltz_1(.5 * L, sigma, A)
    return pre_fact * (I_p - I_m)


###############################################################################


def me_1D_evolver(sol, gpara, params):
    """TODO: Docstring for me_1D_avg_evolver.

    @param sol TODO
    @param gpara TODO
    @param pdict TODO
    @return: TODO
    """
    # Get system parameters
    beta = params['beta']
    ui = params['ui']
    uj = params['uj']
    # N = params['rod_dense']
    # P = params['polarity']
    L = params['L']
    # Get motor parameters
    co = params['co']
    vo = params['vo']
    ks = params['ks']
    fs = params['fs']
    ko = params['ko']
    kappa = float(vo * ks / fs)
    # Get source terms
    q_arr = get_src_term_arr(sol[0], ui, uj, L, co, ks, beta)
    # print("Source terms", q_arr)
    # Evolve positions
    dx_arr = dx_dt(sol[0], ui, uj, sol[1:], gpara, ks)
    # Evolve moments
    dmu_arr = calc_1D_moment_derivs(sol[0], ui, uj, sol[1:], q_arr,
                                    ko, vo, kappa)
    dsol = np.concatenate(([dx_arr], dmu_arr))
    print(dsol)
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError('Infinity or NaN thrown in ODE solver derivatives. '
                           'Current derivatives', dsol)
    return dsol


def create_me_1D_evolver(slvr, sol_init):
    """!Create a closure for ode solver

    @param slvr: MomentExpansion1DAvgSolver solver class
    @param sol_init: Array of time - dependent variables in the ODE
    @return: evolver function for ODE of interest

    """
    gpara, _, _ = get_rod_drag_coeff(slvr.visc, slvr.L, slvr.rod_diam)

    def me_1D_evolver_closure(t, sol):
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(
                ('Infinity or NaN thrown in ODE solver solutions.'
                 ' Current solution: '), sol)
        return me_1D_evolver(sol, gpara, slvr.__dict__)
    return me_1D_evolver_closure


##########################################
