#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad, dblquad
from math import erf, exp, log
from numba import jit, njit

"""@package docstring
File: ME_gen_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

###################################
#  Boltzmann factor calculations  #
###################################


@njit
def boltz_fact_gen(s_i, s_j, rsqr, a_ij, a_ji, b, ks, ho, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two rods

    @param s_i: Position of a bound motor end on rod_i relative to the rods center
    @param s_j: Position of a bound motor end on rod_i relative to the rods center
    @param rsqr: Magnitude squared of the vector from rod_i's COM to rod_j's COM
    @param a_ij: Dot product of u_i and r_ij
    @param a_ji: Dot product of u_j and r_ij
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: Computed Boltzmann factor

    """
    return np.exp(-.5 * beta * ks * np.power(
        np.sqrt(rsqr + s_i**2 + s_j**2 -
                (2. * s_i * s_j * b) +
                2. * (s_j * a_ji - s_i * a_ij)) - ho, 2))


@njit
def weighted_boltz_fact_gen(
        s_i, s_j, pow_i, pow_j, rsqr, a_ij, a_ji, b, ks, ho, beta):
    """!Boltzmann factor for a zero rest length crosslinking motor bound to two
    rods multiplied by s_i and s_j raised to specified powers

    @param s_i: Position of a bound motor end on rod_i relative to the rods center
    @param s_j: Position of a bound motor end on rod_i relative to the rods center
    @param pow1: Power of s_i to weight Boltzmann factor by
    @param pow2: Power of s_j to weight Boltzmann factor by
    @param rsqr: Magnitude squared of the vector from rod_i's COM to rod_j's COM
    @param a_ij: Dot product of u_i and r_ij
    @param a_ji: Dot product of u_j and r_ij
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @return: TODO

    """
    return (np.power(s_i, pow_i) * np.power(s_j, pow_j) *
            boltz_fact_gen(s_i, s_j, rsqr, a_ij, a_ji, b, ks, ho, beta))


############################################
#  Semi-anti derivatives for source terms  #
############################################
sqrt_pi = np.sqrt(np.pi)  # Reduce the number of sqrts you need to do


@njit
def avg_force_gen(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b, mu00, mu10, mu01,
                  mu11, mu20, mu02, mu21, mu12, mu30, mu03, ks, ho):
    """!Find the average force of zero rest length (gen) crosslinkers on rods

    @param r_ij: Vector from the center of mass of rod_i to the center of mass of rod_j
    @param u_i: Orientation unit vector of rod_i
    @param u_j: Orientation unit vector of rod_j
    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s_i
    @param mu01: First motor moment of s_j
    @param ks: motor spring constant
    return: Vector of force from rod_i on rod_j

    """
    drh2 = rsqr - (ho * ho)
    return (-ks * ((-drh2 * mu00 + 2. * (a_ij * mu10 - a_ji * mu01 + b * mu11) - mu20 - mu02) * r_ij
                   + (drh2 * mu10 + 2. * (a_ji * mu11 - a_ij *
                                          mu20 - b * mu21) + mu12 + mu30) * u_i
                   + (-drh2 * mu01 + 2. * (a_ij * mu11 - a_ji * mu02 + b * mu12) - mu21 - mu03) * u_j))


@njit
def avg_force_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
                       mu00, mu10, mu01, mu11, mu20, mu02, ks, ho):
    """!Find the average force of zero rest length (gen) crosslinkers on rods

    @param r_ij: Vector from the center of mass of rod_i to the center of mass of rod_j
    @param u_i: Orientation unit vector of rod_i
    @param u_j: Orientation unit vector of rod_j
    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s_i
    @param mu01: First motor moment of s_j
    @param ks: motor spring constant
    @param ho: motor spring rest length
    return: Vector of force from rod_i on rod_j

    """
    drh2 = rsqr - (ho * ho)

    return -ks * ((-drh2 * mu00 + 2. * (a_ij * mu10 - a_ji * mu01 + b * mu11) - mu20 - mu02) * r_ij
                  + (drh2 * mu10 + 2. * (a_ji * mu11 - a_ij * mu20)) * u_i
                  + (-drh2 * mu01 + 2. * (a_ij * mu11 - a_ji * mu02)) * u_j)
