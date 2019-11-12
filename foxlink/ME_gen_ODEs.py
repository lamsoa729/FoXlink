#!/usr/bin/env python

"""@package docstring
File: ME_gen_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from scipy.integrate import quad, dblquad
# import numpy as np
from numba import njit
from .ME_gen_helpers import (boltz_fact_gen, weighted_boltz_fact_gen,
                             avg_force_gen, avg_force_gen_2ord)


@njit
def du_dt_gen(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
              mu10, mu11, mu20, mu21, mu12, mu30, mu22, mu31, mu13, mu40,
              ks, ho, grot):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length. (Notation is consistent with paper and thesis.)

    @param r_ij: Vector from rod_i's center of mass to rod_j's center of mass
    @param u_i: Orientation unit vector of rod1
    @param u_j: Orientation unit vector of rod2
    @param mu{kl}: motor momement of s_i^k,s_j^l
    @param a_ij: dot product of r_ij and u1
    @param a_ji: dot product of -r_ij and u2
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param ho: motor rest length
    @param grot: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot) * (
        (drh2 * mu10 - 2. * (a_ji * mu11 + a_ij * mu20 + b * mu21) + mu12 + mu30) * r_ij
        - (a_ij * mu10 + b * mu11 + (drh2 - 1.) * mu20
           + 2. * (a_ji * mu21 + a_ij * mu30 + b * mu31) + mu22 + mu40) * u_i
        + (drh2 * mu11 - 2. * (a_ji * mu12 + a_ij * mu21 + b * mu22) + mu31 + mu13) * u_j)


@njit
def du_dt_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b, mu10, mu11, mu20,
                   ks, ho, grot):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length. (Notation is consistent with paper and thesis)

    TODO

    @return: Time-derivative of rod_i's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot) * ((drh2 * mu10 - 2. * (a_ji * mu11 + a_ij * mu20)) * r_ij
                          - (a_ij * mu10 + b * mu11 + (drh2 - 1.) * mu20) * u_i
                          + (drh2 * mu11) * u_j)


@njit
def dmu00_dt_gen_2ord(rsqr, a_ij, a_ji, b, mu00,
                      ko, co, ks, ho, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    @return:
    """
    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return ko * (q - mu00)


@njit
def dmu10_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu00, mu10, mu01, mu11, mu20, mu02,
                      ko, co, vo, fs, ks, ho, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    @return:
    """
    # Redefine some parameters
    kappa = .5 * ks * vo / (ho * ho * fs)
    vo_new = 2. * ho * ho * fs / ks
    ko_new = 2. * ko * ho * ho * fs / (vo * ks)
    drh2 = rsqr - (ho * ho)

    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return


@njit
def dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu10, mu01, mu11, mu20, mu02,
                      ko, co, vo, fs, ks, ho, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    @return:
    """
    # Redefine some parameters
    kappa = .5 * ks * vo / (ho * ho * fs)
    vo_new = 2. * ho * ho * fs / ks
    ko_new = 2. * ko * ho * ho * fs / (vo * ks)
    drh2 = rsqr - (ho * ho)

    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return


@njit
def dmu20_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu10, mu11, mu20,
                      ko, co, vo, fs, ks, ho, beta, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    @return:
    """
    # Redefine some parameters
    kappa = .5 * ks * vo / (ho * ho * fs)
    vo_new = 2. * ho * ho * fs / ks  # vo/kappa
    ko_new = 2. * ko * ho * ho * fs / (vo * ks)  # ko/kappa
    drh2 = rsqr - (ho * ho)

    if q is None:
        #  TODO: Get q <08-11-19, ARL> #
        q = 0
    return
