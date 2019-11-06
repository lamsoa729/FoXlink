#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad, dblquad
from math import erf, exp, log
from numba import jit, njit

"""@package docstring
File: ME_frl_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

###################################
#  Boltzmann factor calculations  #
###################################


@njit
def boltz_fact(s1, s2, rsqr, a1, a2, b, ks, ho, beta):
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
    return np.exp(-.5 * beta * ks * np.power(
        np.sqrt(rsqr + s1**2 + s2**2 -
                (2. * s1 * s2 * b) +
                2. * (s2 * a2 - s1 * a1)) - ho, 2))


@njit
def weighted_boltz_fact(s1, s2, pow1, pow2, rsqr, a1, a2, b, ks, ho, beta):
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
            boltz_fact(s1, s2, rsqr, a1, a2, b, ks, ho, beta))


############################################
#  Semi-anti derivatives for source terms  #
############################################
sqrt_pi = np.sqrt(np.pi)  # Reduce the number of sqrts you need to do


@njit
def avg_force_frl(r12, u1, u2, rsqr, a1, a2, b, rho, P1, P2,
                  mu11, mu20, mu02, mu21, mu12, mu30, mu03, ks, ho):
    """!Find the average force of zero rest length (frl) crosslinkers on rods

    @param r12: Vector from the center of mass of rod1 to the center of mass of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param ks: motor spring constant
    return: Vector of force from rod1 on rod2

    """
    dhr2 = (ho * ho) - rsqr
    return (-ks * ((dhr2 * rho + 2. * (a1 * P1 - a2 * P2 + b * mu11) - mu20 - mu02) * r12
        + (-dhr2 * P1 + 2. * (a2 * mu11 - a1 * mu20 - b * mu21) + mu12 + mu30) * u1
        + (dhr2 * P2 + 2. * (a1 * mu11 - a2 * mu02 + b * mu12) - mu21 - mu03) * u2))


@njit
def avg_force_frl_2order(r12, u1, u2, rsqr, a1, a2, b,
                         rho, P1, P2, mu11, mu20, mu02, ks, ho):
                  """!Find the average force of zero rest length (frl) crosslinkers on rods

    @param r12: Vector from the center of mass of rod1 to the center of mass of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param ks: motor spring constant
    return: Vector of force from rod1 on rod2

    """
    dhr2 = (ho * ho) - rsqr
    return -ks * ((dhr2 * rho + 2. * (a1 * P1 - a2 * P2 + b * mu11) - mu20 - mu02) * r12  
            + (-dhr2 * P1 + 2. * (a2 * mu11 - a1 * mu20 )) * u1 
            + (dhr2 * P2 + 2. * (a1 * mu11 - a2 * mu02 )) * u2)

