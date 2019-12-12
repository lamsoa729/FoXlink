#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad, dblquad
from math import erf, exp, log
from numba import jit, njit

"""@package docstring
File: ME_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

###################################
#  Boltzmann factor calculations  #
###################################


@njit
def boltz_fact_zrl(s1, s2, rsqr, a1, a2, b, ks, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two rods

    @param s1: Position of a bound motor end on rod1 relative to the rods center
    @param s2: Position of a bound motor end on rod1 relative to the rods center
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: Computed Boltzmann factor

    """
    return np.exp(-.5 * beta * ks * (rsqr + s1**2 + s2**2 -
                                     (2. * s1 * s2 * b) +
                                     2. * (s2 * a2 - s1 * a1)))


@njit
def weighted_boltz_fact_zrl(s1, s2, pow1, pow2, rsqr, a1, a2, b, ks, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two
    rods multiplied by s1 and s2 raised to specified powers

    @param s1: Position of a bound motor end on rod1 relative to the rods center
    @param s2: Position of a bound motor end on rod1 relative to the rods center
    @param pow1: Power of s1 to weight Boltzmann factor by
    @param pow2: Power of s2 to weight Boltzmann factor by
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: TODO

    """
    return (np.power(s1, pow1) * np.power(s2, pow2) *
            np.exp(-.5 * beta * ks * (rsqr + s1**2 + s2**2 -
                                      (2. * s1 * s2 * b) +
                                      2. * (s2 * a2 - s1 * a1))))


############################################
#  Semi-anti derivatives for source terms  #
############################################
sqrt_pi = np.sqrt(np.pi)  # Reduce the number of sqrts you need to do


@njit
def semi_anti_deriv_boltz_0(L, s1, sigma, A):
    """!Fast calculation of the s2 integral of the source term for the zeroth
    moment.

    @param L: minus or plus end of bound
    @param s1: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b s1
    @return: One term in the anti-derivative of the boltzman factor integrated over s2

    """
    return (.5 * sqrt_pi * sigma) * erf((L + A) / sigma)


@njit
def semi_anti_deriv_boltz_1(L, s1, sigma, A):
    """!Fast calculation of the s2 integral of the source term for the first
    moment.

    @param L: minus or plus end of bound
    @param s1: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b s1
    @return: One term in the anti-derivative of the boltzman factor integrated over s2

    """
    B = (L + A) / sigma
    return (-.5 * sigma) * (sigma * np.exp(-1. * B * B) +
                            (A * sqrt_pi * erf(B)))


@njit
def semi_anti_deriv_boltz_2(L, s1, sigma, A):
    """!Fast calculation of the s2 integral of the source term for the second
    moment.

    @param L: minus or plus end of bound
    @param s1: location along the first rod
    @param sigma: sqrt(2 kBT/crosslinker spring constant)
    @param A: a2 - b*s1
    @return: One term in the anti-derivative of the boltzman factor integrated over s2

    """
    # inv_sig = 1. / sigma
    B = (L + A) / sigma
    return (.25 * sigma) * (2. * sigma * (A - L) * np.exp(-1. * B * B) +
                            (((2. * A * A) + (sigma * sigma)) * sqrt_pi) * erf(B))


@njit
def fast_zrl_src_integrand_k0(s1, L2, a1, a2, b, sigma, log_pre_fact=0, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s1: TODO
    @param L2: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param sigma: TODO
    @param pre_fact: TODO
    @param k: TODO
    @return: TODO

    """
    A = a2 - (b * s1)
    log_pre_fact -= (s1 * (s1 - 2. * a1) - (A * A)) / (sigma * sigma)

    pre_fact = np.power(s1, k) * np.exp(log_pre_fact)
    # ((s1 * (s1 - 2. * a1)) - (A * A)) / (sigma * sigma))
    I_m = semi_anti_deriv_boltz_0(-.5 * L2, s1, sigma, A)
    I_p = semi_anti_deriv_boltz_0(.5 * L2, s1, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_k1(s1, L2, a1, a2, b, sigma, log_pre_fact=0, k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k1.

    @param s1: TODO
    @param L2: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param sigma: TODO
    @param pre_fact: TODO
    @param k: TODO
    @return: TODO

    """
    A = a2 - (b * s1)
    log_pre_fact -= (s1 * (s1 - 2. * a1) - (A * A)) / (sigma * sigma)
    pre_fact = np.power(s1, k) * np.exp(log_pre_fact)

    # pre_fact = np.power(s1, k) * np.exp(log_pre_fact)

    # pre_fact *= pow(s1, k) * exp(-1. *
    # ((s1 * (s1 - 2. * a1)) - (A * A)) / (sigma * sigma))
    I_m = semi_anti_deriv_boltz_1(-.5 * L2, s1, sigma, A)
    I_p = semi_anti_deriv_boltz_1(.5 * L2, s1, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_k2(s1, L2, a1, a2, b, sigma, log_pre_fact=0., k=0):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s1: TODO
    @param L2: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    A = a2 - (b * s1)
    log_pre_fact -= (s1 * (s1 - 2. * a1) - (A * A)) / (sigma * sigma)
    pre_fact = np.power(s1, k) * np.exp(log_pre_fact)
    # pre_fact *= np.power(s1, k) * np.exp(-1. *
    # ((s1 * (s1 - 2. * a1)) - (A * A)) / (sigma * sigma))
    I_m = semi_anti_deriv_boltz_2(-.5 * L2, s1, sigma, A)
    I_p = semi_anti_deriv_boltz_2(.5 * L2, s1, sigma, A)
    return pre_fact * (I_p - I_m)


def fast_zrl_src_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=0, l=0):
    """!TODO: Docstring for fast_zrl_src_kl

    @param s1: TODO
    @param L2: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param sigma: TODO
    @param l: TODO
    @return: TODO

    """
    if l == 0:
        integrand = fast_zrl_src_integrand_k0
    elif l == 1:
        integrand = fast_zrl_src_integrand_k1
    elif l == 2:
        integrand = fast_zrl_src_integrand_k2
    else:
        raise RuntimeError(
            "{}-order derivatives have not been implemented for fast source solver.".format(l))
    sigma = np.sqrt(2. / (ks * beta))
    log_pre_fact = -.5 * rsqr * ks * beta
    q, e = quad(integrand, -.5 * L1, .5 * L1,
                args=(L2, a1, a2, b, sigma, log_pre_fact, k))
    return q


@njit
def avg_force_zrl(r12, u1, u2, rho, P1, P2, ks):
    """!Find the average force of zero rest length (zrl) crosslinkers on rods

    @param r12: Vector from the center of mass of rod1 to the center of mass of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param ks: motor spring constant
    return: Vector of force from rod1 on rod2

    """
    return -ks * (r12 * rho + P2 * u2 - P1 * u1)
