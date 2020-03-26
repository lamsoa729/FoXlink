#!/usr/bin/env python
"""@package docstring
File: me_zrl_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from math import erf
from numba import njit
from scipy.integrate import quad
from .me_helpers import convert_sol_to_geom


def get_zrl_moments(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    return sol[12:18].tolist()


def get_zrl_moments_and_boundary_terms(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    return (sol[12:18].tolist(), sol[18:26].tolist())

###################################
#  Boltzmann factor calculations  #
###################################


@njit
def boltz_fact_zrl(s_i, s_j, rsqr, a1, a2, b, ks, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two rods

    @param s_i: Position of a bound motor end on rod1 relative to the rods center
    @param s_j: Position of a bound motor end on rod1 relative to the rods center
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: Computed Boltzmann factor

    """
    return np.exp(-.5 * beta * ks * (rsqr + s_i**2 + s_j**2 -
                                     (2. * s_i * s_j * b) +
                                     2. * (s_j * a2 - s_i * a1)))


@njit
def weighted_boltz_fact_zrl(s_i, s_j, pow1, pow2, rsqr, a1, a2, b, ks, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two
    rods multiplied by s_i and s_j raised to specified powers

    @param s_i: Position of a bound motor end on rod1 relative to the rods center
    @param s_j: Position of a bound motor end on rod1 relative to the rods center
    @param pow1: Power of s_i to weight Boltzmann factor by
    @param pow2: Power of s_j to weight Boltzmann factor by
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: TODO

    """
    return (np.power(s_i, pow1) * np.power(s_j, pow2) *
            np.exp(-.5 * beta * ks * (rsqr + s_i**2 + s_j**2 -
                                      (2. * s_i * s_j * b) +
                                      2. * (s_j * a2 - s_i * a1))))


############################################
#  Semi-anti derivatives for source terms  #
############################################
SQRT_PI = np.sqrt(np.pi)  # Reduce the number of sqrts you need to do


@njit
def semi_anti_deriv_boltz_0(L, sigma, A):
    """!Fast calculation of the s_j integral of the source term for the zeroth
    moment.

    @param L: minus or plus end of bound
    @param s_i: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 + b s_i
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    return (.5 * SQRT_PI * sigma) * erf((L + A) / sigma)


@njit
def semi_anti_deriv_boltz_1(L, sigma, A):
    """!Fast calculation of the s_j integral of the source term for the first
    moment.

    @param L: minus or plus end of bound
    @param s_i: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b s_i
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    B = (L + A) / sigma
    return (-.5 * sigma) * (sigma * np.exp(-1. * B * B) +
                            (A * SQRT_PI * erf(B)))


@njit
def semi_anti_deriv_boltz_2(L, sigma, A):
    """!Fast calculation of the s_j integral of the source term for the second
    moment.

    @param L: minus or plus end of bound
    @param s_i: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b*s_i
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    B = (L + A) / sigma
    return (.25 * sigma) * (2. * sigma * (A - L) * np.exp(-1. * B * B) +
                            (((2. * A * A) + (sigma * sigma)) * SQRT_PI) * erf(B))


@njit
def semi_anti_deriv_boltz_3(L, sigma, A):
    """!Fast calculation of the s_j integral of the source term for the second
    moment.

    @param L: minus or plus end of bound
    @param s_i: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b*s_i
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    B = (L + A) / sigma
    return (-.25 * sigma) * ((2. * sigma * (A * A - A * L + L * L + sigma * sigma)
                              * np.exp(-1. * B * B))
                             + ((2. * A * A) + 3. * (sigma * sigma))
                             * A * SQRT_PI * erf(B))


@njit
def fast_zrl_src_integrand_l0(
        s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    A = -1. * (a_ji + (b * s_i))
    exponent = -1. * (rsqr + s_i * (s_i - 2. * a_ij) -
                      (A * A)) / (sigma * sigma)

    pre_fact = np.power(s_i, k) * np.exp(exponent)
    # ((s_i * (s_i - 2. * a1)) - (A * A)) / (sigma * sigma))
    I_m = semi_anti_deriv_boltz_0(-.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_0(.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_l1(
        s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k1.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    A = -1. * (a_ji + (b * s_i))
    exponent = -1. * (rsqr + s_i * (s_i - 2. * a_ij) -
                      (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_1(-.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_1(.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_l2(
        s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    A = -1. * (a_ji + (b * s_i))
    exponent = -1. * (rsqr + s_i * (s_i - 2. * a_ij) -
                      (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    # pre_fact *= np.power(s_i, k) * np.exp(-1. *
    # ((s_i * (s_i - 2. * a1)) - (A * A)) / (sigma * sigma))
    I_m = semi_anti_deriv_boltz_2(-.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_2(.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


def fast_zrl_src_integrand_l3(
        s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    A = -1. * (a_ji + (b * s_i))
    exponent = -1. * (rsqr + s_i * (s_i - 2. * a_ij) -
                      (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_3(-.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_3(.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


def fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=0):
    """!TODO: Docstring for fast_zrl_src_kl

    @param s_i: TODO
    @param L2: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param sigma: TODO
    @param l: TODO
    @return: TODO

    """
    if l == 0:
        integrand = fast_zrl_src_integrand_l0
    elif l == 1:
        integrand = fast_zrl_src_integrand_l1
    elif l == 2:
        integrand = fast_zrl_src_integrand_l2
    else:
        raise RuntimeError(
            "{}-order derivatives have not been implemented for fast source solver.".format(l))
    sigma = np.sqrt(2. / (ks * beta))
    q, e = quad(integrand, -.5 * L_i, .5 * L_i,
                args=(L_j, rsqr, a_ij, a_ji, b, sigma, k))
    return q

########################################
#  Preparation functions for evolvers  #
########################################


@njit
def get_Qj_params(s_i, L_j, a_ji, b, ks, beta):
    hL_j = .5 * L_j
    sigma = np.sqrt(2. / (ks * beta))
    A_j = -1. * (a_ji + (b * s_i))
    return hL_j, sigma, A_j


def prep_zrl_evolver(sol, params):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param arg1: TODO
    @return: TODO

    """
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    c = params['co']
    L_i, L_j = params['L1'], params['L2']
    ks = params['ks']
    beta = params['beta']

    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1. * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    q00 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=0)
    q10 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji,
                              a_ij, b, ks, beta, k=0, l=1)
    q01 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=1)
    q11 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=1, l=1)
    q20 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji,
                              a_ij, b, ks, beta, k=0, l=2)
    q02 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=2)
    return [rsqr, a_ij, a_ji, b], [q00, q10, q01, q11, q20, q02]


def prep_zrl_bound_evolver(sol, params):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param sol: TODO
    @param params: TODO
    @return: TODO

    """
    c = params['co']
    L_i, L_j = params['L1'], params['L2']
    ks = params['ks']
    beta = params['beta']

    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    (rsqr, a_ij, a_ji, b) = scalar_geom

    hL_j, sigma, A_j = get_Qj_params(.5 * L_i, L_j, a_ji, b, ks, beta)
    hL_i, sigma, A_i = get_Qj_params(hL_j, L_i, a_ij, b, ks, beta)
    Q0_j = c * fast_zrl_src_integrand_l0(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q0_i = c * fast_zrl_src_integrand_l0(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q1_j = c * fast_zrl_src_integrand_l1(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q1_i = c * fast_zrl_src_integrand_l1(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q2_j = c * fast_zrl_src_integrand_l2(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q2_i = c * fast_zrl_src_integrand_l2(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q3_j = c * fast_zrl_src_integrand_l3(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q3_i = c * fast_zrl_src_integrand_l3(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    return (scalar_geom, q_arr,
            [Q0_j, Q0_i, Q1_j, Q1_i, Q2_j, Q2_i, Q3_j, Q3_i])


@njit
def avg_force_zrl(r_ij, u_i, u_j, mu00, mu10, mu01, ks):
    """!Find the average force of zero rest length (zrl) crosslinkers on rods

    @param r12: Vector from the center of mass of rod1 to the center of mass of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s_i
    @param P2: First motor moment of s_j
    @param ks: motor spring constant
    return: Vector of force from rod1 on rod2

    """
    return -ks * (r_ij * mu00 + mu01 * u_j - mu10 * u_i)
