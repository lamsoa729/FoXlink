#!/usr/bin/env python

"""@package docstring
File: me_gen_odes.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from scipy.integrate import quad, dblquad
import numpy as np
from numba import njit
from .me_gen_helpers import (boltz_fact_gen, weighted_boltz_fact_gen,
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
def du_dt_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
                   mu10, mu11, mu20,
                   ks, ho, grot):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length. (Notation is consistent with paper and thesis)

    Test symmetry with asymmetric geometry but symmetric distribution
    # >>> r_ij, u_i, u_j = (np.asarray([1,2,3]), np.asarray([1,0,0]), np.asarray([0,1,0]))
    # >>> rsqr, a_ij, a_ji, b = (np.dot(r_ij, r_ij), np.dot(u_i, r_ij), np.dot(u_j, -1.*r_ij), np.dot(u_j,u_i))
    # >>> mu10, mu01, mu11, mu20, mu02, ks, ho, grot = (2,2,3,1,1,4,5,6)
    # >>> c1 = du_dt_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b, mu10, mu11, mu20,
    # ... ks, ho, grot)
    # >>> c2 = du_dt_gen_2ord(-1.*r_ij, u_j, u_i, rsqr, a_ji, a_ij, b, mu01, mu11, mu02,
    # ... ks, ho, grot)

    @return: Time-derivative of rod_i's orientation vector
    """
    drh2 = rsqr - (ho * ho)
    return (ks / grot) * ((drh2 * mu10 - 2. * (a_ji * mu11 + a_ij * mu20)) * r_ij
                          - (a_ij * mu10 + b * mu11 + (drh2 - 1.) * mu20) * u_i
                          + (drh2 * mu11) * u_j)


@njit
def dmu00_dt_gen(mu00, ko, q=0):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    @return:
    """
    return ko * (q - mu00)


@njit
def dmu10_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu00, mu10, mu01, mu11, mu20, mu02,
                      ko, vo, fs, ks, ho, q):
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

    return ko * q + kappa * (a_ij * (drh2 + vo_new) * mu00
                             + (b * drh2 - 2. * a_ij * a_ji) * mu01
                             - (2. * a_ij * a_ij + drh2 + ko_new) * mu10
                             + (2. * a_ji - 4. * a_ij * b) * mu11
                             + (a_ij - 2. * a_ji * b) * mu02
                             + 3. * a_ij * mu20)


@njit
def dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu10, mu01, mu11, mu20, mu02,
                      ko, vo, fs, ks, ho, q=None):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    TODO

    -----------------------
    Antiparallel case. Should be symmetric
    >>> (rsqr, a_ij, a_ji, b, mu10, mu01, mu11, mu20, mu02, ko, vo, fs, ks, ho, q) = (1,0,0,-1,1,1,1,1,1,0,2,2,1,2,0)
    >>> c1 = dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b, mu10, mu01, mu11, mu20,
    ... mu02, ko, vo, fs, ks, ho, q)
    >>> c2 = dmu11_dt_gen_2ord(rsqr, a_ji, a_ij, b,  mu01, mu10, mu11, mu02,
    ... mu20, ko, vo, fs, ks, ho, q)
    >>> c1 == c2
    True

    Test symmetry with asymmetric case
    >>> (rsqr, a_ij, a_ji, b, mu10, mu01, mu11, mu20, mu02, ko, vo, fs, ks, ho, q) = (
    ... 1,2,3,-1,4,5,6,7,8,9,10,11,12,13,14)
    >>> c1 = dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b, mu10, mu01, mu11, mu20,
    ... mu02, ko, vo, fs, ks, ho, q)
    >>> c2 = dmu11_dt_gen_2ord(rsqr, a_ji, a_ij, b,  mu01, mu10, mu11, mu02,
    ... mu20, ko, vo, fs, ks, ho, q)
    >>> c1 == c2
    True
    >>> # Make sure it fails when not ij->ji and kl->lk is not satisfied
    >>> c3 = dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b,  mu01, mu10, mu11, mu02,
    ... mu20, ko, vo, fs, ks, ho, q)
    >>> c1 == c3
    False

    @return: Derivative of the mu_ij^11 moment.
    """
    # Redefine some parameters
    kappa = .5 * ks * vo / (ho * ho * fs)
    vo_new = 2. * ho * ho * fs / ks
    ko_new = 2. * ko * ho * ho * fs / (vo * ks)
    drh2 = rsqr - (ho * ho)

    return ko * q + kappa * ((a_ij * drh2 + vo_new) * mu01
                             + (a_ji * drh2 + vo_new) * mu10
                             - 2. * (a_ij * a_ij + a_ji * a_ji +
                                     drh2 + .5 * ko_new) * mu11
                             + (b * drh2 - 2. * a_ij * a_ji) * (mu02 * mu20))


@njit
def dmu20_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                      mu10, mu11, mu20,
                      ko, vo, fs, ks, ho, q=None):
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

    return ko * q + kappa * ((2. * a_ij * drh2 + vo_new) * mu10
                             + (2. * drh2 * b - 4. * a_ij * a_ji) * mu11
                             - (4. * a_ij * a_ij + 2. * drh2 + ko_new) * mu20)
