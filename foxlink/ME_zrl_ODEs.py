#!/usr/bin/env python

"""@package docstring
File: ME_zrl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Class that contains the all ODEs relevant to solving the moment
expansion formalism of the Fokker-Planck equation for bound crosslinking motors.
"""
from scipy.integrate import dblquad
from numba import njit
from .ME_zrl_helpers import (boltz_fact_zrl, weighted_boltz_fact_zrl,
                             fast_zrl_src_full_kl)


@njit
def du1_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot1):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r_ij: Vector from rod1's center of mass to rod2's center of mass
    @param u_i: Orientation unit vector of rod1
    @param u_j: Orientation unit vector of rod2
    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a_ij: dot product of r_ij and u_i
    @param b: dot product of u_i and u_j
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    return (ks / grot1) * ((r_ij - (a_ij * u_i))
                           * mu10 + (u_j - (b * u_i)) * mu11)


@njit
def du2_dt_zrl(r_ij, u_i, u_j, mu01, mu11, a2, b, ks, grot2):
    """!Calculate the time-derivative of rod2's orientation vector with respect
    to the current state of the crosslinked rod system when motor have
    zero rest length.

    @param r_ij: Vector from rod1's center of mass to rod2's center of mass
    @param u_i: Orientation unit vector of rod1
    @param u_j: Orientation unit vector of rod2
    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a_ij: Dot product of r_ij and u_i
    @param b: Dot product of u_i and u_j
    @param ks: Motor spring constant
    @param grot2: Rotational drag coefficient of rod2
    @return: Time-derivative of rod2's orientation vector
    """
    return (ks / grot2) * (((a2 * u_j) - r_ij)
                           * mu01 + (u_i - (b * u_j)) * mu11)

################################
#  Moment evolution functions  #
################################


def dmu00_dt_zrl(mu00, rsqr, a_ij, a2, b, vo, fs, ko,
                 c, ks, beta, L_i, L_j, q00='fast'):
    """!Calculate the time-derivative of the zeroth moment of the zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @param q: Binding source term (i.e. partition function)
    @return: Time derivative of the zeroth moment of motors

    """
    # Partition function
    if q00 is None:
        q00, e = dblquad(boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[rsqr, a_ij, a2, b, ks, beta])
    elif q00 == 'fast':
        q00 = fast_zrl_src_full_kl(
            L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=0, l=0)

    return ko * (c * q00 - mu00)


def dmu10_dt_zrl(mu00, mu10, mu01, rsqr, a_ij, a2, b, vo, fs,
                 ko, c, ks, beta, L_i, L_j, q10='fast'):
    """!Calculate the time-derivative of the first moment(s1) of the zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @param q10: Binding source term of first moment
    @return: Time derivative of the first(s1) moment of motors

    """
    # Partition function
    if q10 is None:
        q10, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[1, 0, rsqr, a_ij, a2, b, ks, beta],)
    elif q10 == 'fast':
        # q10 = fast_zrl_src_full_kl(L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=1, l=0)
        # Make coordinate transformation
        q10 = fast_zrl_src_full_kl(
            L_j, L_i, rsqr, -a2, -a_ij, b, ks, beta, k=0, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q10) + ((vo + kappa * a_ij) * mu00) - ((ko + kappa) * mu10)
            + (kappa * b * mu01))


def dmu01_dt_zrl(mu00, mu10, mu01, rsqr, a_ij, a2, b, vo,
                 fs, ko, c, ks, beta, L_i, L_j, q01='fast'):
    """!Calculate the time-derivative of the first moment(s2) of zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @return: Time derivative of the first(s2) moment of motors

    """
    # Partition function
    if q01 is None:
        q01, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[0, 1, rsqr, a_ij, a2, b, ks, beta])
    elif q01 == 'fast':
        q01 = fast_zrl_src_full_kl(
            L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=0, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q01) + ((vo - kappa * a2) * mu00) - ((ko + kappa) * mu01)
            + (kappa * b * mu10))


def dmu11_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                 a_ij, a2, b, vo, fs, ko, c, ks, beta, L_i, L_j, q11='fast'):
    """!Calculate the time-derivative of the second moment(s1,s2) of zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r_ij)
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @return: Time derivative of the second(s1,s2) moment of motors

    """
    # Partition function
    if q11 is None:
        q11, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[1, 1, rsqr, a_ij, a2, b, ks, beta])
    elif q11 == 'fast':
        q11 = fast_zrl_src_full_kl(
            L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=1, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q11) + ((vo - kappa * a2) * mu10) + ((vo + kappa * a_ij) * mu01)
            - ((ko + 2. * kappa) * mu11) + (kappa * b * (mu20 + mu02)))


def dmu20_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                 a_ij, a2, b, vo, fs, ko, c, ks, beta, L_i, L_j, q20='fast'):
    """!Calculate the time-derivative of the second moment(s1^2) of zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r_ij)
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @return: Time derivative of the second(s1^2) moment of motors

    """
    # Partition function
    if q20 is None:
        q20, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[2, 0, rsqr, a_ij, a2, b, ks, beta])
    elif q20 == 'fast':
        # q20 = fast_zrl_src_full_kl(L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=2, l=0)
        # Make coordinate transformation
        q20 = fast_zrl_src_full_kl(
            L_j, L_i, rsqr, -a2, -a_ij, b, ks, beta, k=0, l=2)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q20) + (2. * (vo + kappa * a_ij) * mu10)
            + (2. * kappa * b * mu11) - ((ko + 2. * kappa) * mu20))


def dmu02_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                 a_ij, a2, b, vo, fs, ko, c, ks, beta, L_i, L_j, q02='fast'):
    """!Calculate the time-derivative of the second moment(s2^2) of zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r_ij)
    @param a_ij: Dot product of u_i and r_ij
    @param a2: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @return: Time derivative of the second(s2^2) moment of motors

    """
    # Partition function
    if q02 is None:
        q02, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
                         lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
                         args=[0, 2, rsqr, a_ij, a2, b, ks, beta])
    elif q02 == 'fast':
        q02 = fast_zrl_src_full_kl(
            L_i, L_j, rsqr, a_ij, a2, b, ks, beta, k=0, l=2)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q02) + (2. * (vo - kappa * a2) * mu01) +
            (2. * kappa * b * mu11) - ((ko + 2. * kappa) * mu02))
