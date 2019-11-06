#!/usr/bin/env python

from .ME_helpers import dr_dt
"""@package docstring
File: ME_frl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@njit
def du1_dt_frl(r12, u1, u2, rsqr, a1, a2, b,
               P1, mu11, m20, mu21, mu12, mu30, mu22, mu31, mu13, mu40,
               ks, ho, grot1):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot1) * ((drh2 * P1 + 2. *
                            (a2 * mu11 - a1 * mu20 - b * mu21) + mu12 + mu30) * r12
                           + (-a1 * P1 - b * mu11 + (1. - drh2) * mu20 - 2. *
                              (a2 * mu21 + a1 * mu30 + b * mu31) - mu22 - mu40) * u1
                           + (drh2 * mu11 + 2. *
                               (a2 * mu12 - a1 * mu21 - b * mu22) + mu31 + mu13) * u2)


@njit
def du1_dt_frl_2order(r12, u1, u2, rsqr, a1, a2, b, P1, mu11, mu20,
                      ks, ho, grot1):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot1) * ((drh2 * P1 + 2. * (a2 * mu11 - a1 * mu20)) * r12
                           + (-a1 * P1 - b * mu11 + (1. - drh2) * mu20) * u1
                           + (drh2 * mu11) * u2)


@njit
def du2_dt_frl(r12, u1, u2, rsqr, a1, a2, b,
               P2, mu11, m02, mu21, mu12, mu03, mu22, mu31, mu13, mu04,
               ks, ho, grot2):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot2) * ((-drh2 * P2 + 2. *
                            (a1 * mu11 - a2 * mu02 + b * mu12) - mu12 - mu30) * r12
                           + (drh2 * mu11 + 2. *
                              (a2 * mu12 - a1 * mu21 - b * mu22) + mu31 + mu13) * u1
                           + (a2 * P2 - b * mu11 + (1. - drh2) * mu02 - 2. *
                              (-a1 * mu12 - a2 * mu03 + b * mu13) - mu22 - mu04) * u2)


@njit
def du2_dt_frl_2order(r12, u1, u2, rsqr, a1, a2, b, P2, mu11, mu02,
                      ks, ho, grot2):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot2) * ((-drh2 * P2 + 2. * (a1 * mu11 - a2 * mu02)) * r12
                           + (drh2 * mu11) * u1
                           + (a2 * P2 - b * mu11 + (1. - drh2) * mu02) * u2)
