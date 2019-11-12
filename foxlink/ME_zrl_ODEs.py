#!/usr/bin/env python

"""@package docstring
File: ME_zrl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""
from scipy.integrate import dblquad
from numba import njit
from .ME_zrl_helpers import (boltz_fact_zrl, weighted_boltz_fact_zrl,
                             fast_zrl_src_full_kl)


@njit
def du1_dt_zrl(r12, u1, u2, P1, mu11, a1, b, ks, grot1):
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
    return (ks / grot1) * ((r12 - (a1 * u1)) * P1 + (u2 - (b * u1)) * mu11)


@njit
def du2_dt_zrl(r12, u1, u2, P2, mu11, a2, b, ks, grot2):
    """!Calculate the time-derivative of rod2's orientation vector with respect
    to the current state of the crosslinked rod system when motor have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: Dot product of r12 and u1
    @param b: Dot product of u1 and u2
    @param ks: Motor spring constant
    @param grot2: Rotational drag coefficient of rod2
    @return: Time-derivative of rod2's orientation vector
    """
    return (ks / grot2) * (((a2 * u2) - r12) * P2 + (u1 - (b * u2)) * mu11)

################################
#  Moment evolution functions  #
################################


def drho_dt_zrl(rho, rsqr, a1, a2, b, vo, fs, ko,
                c, ks, beta, L1, L2, q='fast'):
    """!Calculate the time-derivative of the zeroth moment of the zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param q: Binding source term (i.e. partition function)
    @return: Time derivative of the zeroth moment of motors

    """
    # Partition function
    if q is None:
        q, e = dblquad(boltz_fact_zrl, -.5 * L1, .5 * L1,
                       lambda s2: -.5 * L2, lambda s2: .5 * L2,
                       args=[rsqr, a1, a2, b, ks, beta])
    elif q == 'fast':
        q = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=0, l=0)

    return ko * (c * q - rho)


def dP1_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo, fs,
               ko, c, ks, beta, L1, L2, q10='fast'):
    """!Calculate the time-derivative of the first moment(s1) of the zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param q10: Binding source term of first moment
    @return: Time derivative of the first(s1) moment of motors

    """
    # Partition function
    if q10 is None:
        q10, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                         lambda s2: -.5 * L2, lambda s2: .5 * L2,
                         args=[1, 0, rsqr, a1, a2, b, ks, beta],)
    elif q10 == 'fast':
        # q10 = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=1, l=0)
        # Make coordinate transformation
        q10 = fast_zrl_src_full_kl(
            L2, L1, rsqr, -a2, -a1, b, ks, beta, k=0, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q10) + ((vo + kappa * a1) * rho) - ((ko + kappa) * P1)
            + (kappa * b * P2))


def dP2_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo,
               fs, ko, c, ks, beta, L1, L2, q01='fast'):
    """!Calculate the time-derivative of the first moment(s2) of zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @return: Time derivative of the first(s2) moment of motors

    """
    # Partition function
    if q01 is None:
        q01, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                         lambda s2: -.5 * L2, lambda s2: .5 * L2,
                         args=[0, 1, rsqr, a1, a2, b, ks, beta])
    elif q01 == 'fast':
        q01 = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=0, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q01) + ((vo - kappa * a2) * rho) - ((ko + kappa) * P2)
            + (kappa * b * P1))


def dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q11='fast'):
    """!Calculate the time-derivative of the second moment(s1,s2) of zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r12)
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @return: Time derivative of the second(s1,s2) moment of motors

    """
    # Partition function
    if q11 is None:
        q11, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                         lambda s2: -.5 * L2, lambda s2: .5 * L2,
                         args=[1, 1, rsqr, a1, a2, b, ks, beta])
    elif q11 == 'fast':
        q11 = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=1, l=1)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q11) + ((vo - kappa * a2) * P1) + ((vo + kappa * a1) * P2)
            - ((ko + 2. * kappa) * mu11) + (kappa * b * (mu20 + mu02)))


def dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q20='fast'):
    """!Calculate the time-derivative of the second moment(s1^2) of zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r12)
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @return: Time derivative of the second(s1^2) moment of motors

    """
    # Partition function
    if q20 is None:
        q20, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                         lambda s2: -.5 * L2, lambda s2: .5 * L2,
                         args=[2, 0, rsqr, a1, a2, b, ks, beta])
    elif q20 == 'fast':
        # q20 = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=2, l=0)
        # Make coordinate transformation
        q20 = fast_zrl_src_full_kl(
            L2, L1, rsqr, -a2, -a1, b, ks, beta, k=0, l=2)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q20) + (2. * (vo + kappa * a1) * P1)
            + (2. * kappa * b * mu11) - ((ko + 2. * kappa) * mu20))


def dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q02='fast'):
    """!Calculate the time-derivative of the second moment(s2^2) of zero rest
    length crosslinkers bound to rods.

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r12)
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @return: Time derivative of the second(s2^2) moment of motors

    """
    # Partition function
    if q02 is None:
        q02, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                         lambda s2: -.5 * L2, lambda s2: .5 * L2,
                         args=[0, 2, rsqr, a1, a2, b, ks, beta])
    elif q02 == 'fast':
        q02 = fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=0, l=2)
    # Characteristic walking rate
    kappa = vo * ks / fs
    return ((ko * c * q02) + (2. * (vo - kappa * a2) * P2) +
            (2. * kappa * b * mu11) - ((ko + 2. * kappa) * mu02))
