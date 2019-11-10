#!/usr/bin/env python

from .ME_helpers import dr_dt
"""@package docstring
File: ME_frl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@njit
def du1_dt_frl(r12, u1, u2, rsqr, a12, a21, b,
               mu10, mu11, m20, mu21, mu12, mu30, mu22, mu31, mu13, mu40,
               ks, ho, grot):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length. (Notation is consistent with paper and thesis.)

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a12: dot product of r12 and u1
    @param a21: dot product of -r12 and u2
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot) * ((drh2 * mu10 - 2. *
                           (a21 * mu11 + a12 * mu20 + b * mu21) + mu12 + mu30) * r12
                          - (a12 * mu10 + b * mu11 + (drh2 - 1) * mu20 - 2. *
                              (a21 * mu21 + a12 * mu30 + b * mu31) + mu22 + mu40) * u1
                          + (drh2 * mu11 - 2. *
                             (a21 * mu12 + a12 * mu21 + b * mu22) + mu31 + mu13) * u2)


@njit
def du1_dt_frl_2order(r12, u1, u2, rsqr, a12, a21, b, mu10, mu11, mu20,
                      ks, ho, grot):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length. (Notation is consistent with paper and thesis)


    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot) * ((drh2 * mu10 - 2. * (a21 * mu11 + a12 * mu20)) * r12
                          - (a12 * mu10 + b * mu11 + (drh2 - 1.) * mu20) * u1
                          + (drh2 * mu11) * u2)


@njit
def dmu00_dt_frl_2order(rsqr, a12, a21, b, mu00, ko, co, ks, ho, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return ko * (q - mu00)


@njit
def dmu10_dt_frl_2order(rsqr, a12, a21, b, mu00,
                        ko, co, vo, fs, ks, ho, vo, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    # Redefine some parameters
    kappa = .5 * ks * vo / (ho * ho * fs)
    vo_new = 2. * ho * ho * fs / ks
    ko_new = 2. * ko * ho * ho * fs / (vo * ks)
    drh2 = rsqr - (ho * ho)
    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return ko * q + kappa * (0)  # XXXX Finish this up
