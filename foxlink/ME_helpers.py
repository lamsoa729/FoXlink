#!/usr/bin/env python
from scipy.integrate import solve_ivp, quad, dblquad
# from scipy.special import erf
from math import erf, exp, log
import numpy as np
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


@jit
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


@jit
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


def fast_zrl_src_full_kl(L1, L2, rsqr, a1, a2, b, ks, beta, k=0, l=0):
    """!TODO: Docstring for fast_zrl_src_full_kl

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

    # Non-dimensionalize to keep integral stable
    # ND = sigma
    # L1 /= ND
    # L2 /= ND
    # a1 /= ND
    # a2 /= ND
    # sigma /= ND

    q, e = quad(integrand, -.5 * L1, .5 * L1,
                args=(L2, a1, a2, b, sigma, log_pre_fact, k))
    return q
    # return np.power(ND, (2 + l + k)) * q

##################################
#  Geometric evolution functions  #
##################################


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


@njit
def dr_dt_zrl(F, u, gpara, gperp):
    """!Get the evolution of a rods postion given a force, orientation of rod,
    and drag coefficients.

    @param F: Average force exerted on rod
    @param u: Orientation vector of rod
    @param gpara: Parallel friction coefficient of rod
    @param gperp: Perpendicular friction coefficient of rod
    @return: Time-derivative of the rod motion

    """
    mpara = 1. / gpara
    mperp = 1. / gperp
    return (u * (mpara - mperp) * np.dot(F, u)) + (mperp * F)


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


@jit
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
                c, ks, beta, L1, L2, q=None):
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


def dP1_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo,
               fs, ko, c, ks, beta, L1, L2, q10=None):
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
               fs, ko, c, ks, beta, L1, L2, q01=None):
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
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q11=None):
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
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q20=None):
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
                 a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q02=None):
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


#######################
#  Evolver functions  #
#######################

def evolver_zrl(r1, r2, u1, u2,  # Vectors
                rho, P1, P2, mu11, mu20, mu02,  # Moments
                gpara1, gperp1, grot1,  # Friction coefficients
                gpara2, gperp2, grot2,
                vo, fs, ko, c, ks, beta, L1, L2, fast=None):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param r1: Center of mass postion of rod1
    @param r2: Center of mass position of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param gpara1: Parallel drag coefficient of rod1
    @param gperp1: Perpendicular drag coefficient of rod1
    @param grot1: Rotational drag coefficient of rod1
    @param gpara2: Parallel drag coefficient of rod1
    @param gperp2: Perpendicular drag coefficient of rod1
    @param grot2: Rotational drag coefficient of rod1
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param fast: Flag on whether or not to use fast solving techniques
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    r12 = r2 - r1
    rsqr = np.dot(r12, r12)
    a1 = np.dot(r12, u1)
    a2 = np.dot(r12, u2)
    b = np.dot(u1, u2)
    # Get average force of crosslinkers on rod2
    F12 = avg_force_zrl(r12, u1, u2, rho, P1, P2, ks)
    # Evolution of rod positions
    dr1 = dr_dt_zrl(-1. * F12, u1, gpara1, gperp1)
    dr2 = dr_dt_zrl(F12, u2, gpara2, gperp2)
    # Evolution of orientation vectors
    du1 = du1_dt_zrl(r12, u1, u2, P1, mu11, a1, b, ks, grot1)
    du2 = du2_dt_zrl(r12, u1, u2, P2, mu11, a2, b, ks, grot2)
    # Evolution of zeroth moment
    drho = drho_dt_zrl(rho, rsqr, a1, a2, b,
                       vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evoultion of first moments
    dP1 = dP1_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    dP2 = dP2_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu20 = dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu02 = dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dsol = np.concatenate(
        (dr1, dr2, du1, du2, [drho, dP1, dP2, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_ang(u1, u2,  # Vectors
                    rho, P1, P2, mu11, mu20, mu02,  # Moments
                    gpara1, gperp1, grot1,  # Friction coefficients
                    gpara2, gperp2, grot2,
                    r12, rsqr, vo, fs, ko, c, ks, beta, L1, L2,  # Other constants
                    fast=None):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param gpara1: Parallel drag coefficient of rod1
    @param gperp1: Perpendicular drag coefficient of rod1
    @param grot1: Rotational drag coefficient of rod1
    @param gpara2: Parallel drag coefficient of rod1
    @param gperp2: Perpendicular drag coefficient of rod1
    @param grot2: Rotational drag coefficient of rod1
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param fast: Flag on whether or not to use fast solving techniques
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    rod_change_arr = np.zeros(6)
    # Define useful parameters for functions
    a1 = np.dot(r12, u1)
    a2 = np.dot(r12, u2)
    b = np.dot(u1, u2)
    # Get average force of crosslinkers on rod2
    # F12 = avg_force_zrl(r12, u1, u2, rho, P1, P2, ks)
    # Evolution of rod positions
    # dr1 = dr_dt_zrl(-1. * F12, u1, gpara1, gperp1)
    # dr2 = dr_dt_zrl(F12, u2, gpara2, gperp2)
    # Evolution of orientation vectors
    du1 = du1_dt_zrl(r12, u1, u2, P1, mu11, a1, b, ks, grot1)
    du2 = du2_dt_zrl(r12, u1, u2, P2, mu11, a2, b, ks, grot2)
    # Evolution of zeroth moment
    drho = drho_dt_zrl(rho, rsqr, a1, a2, b,
                       vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evoultion of first moments
    dP1 = dP1_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    dP2 = dP2_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu20 = dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu02 = dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dsol = np.concatenate(
        (rod_change_arr, du1, du2, [drho, dP1, dP2, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_orient(r1, r2, u1, u2,  # Vectors
                       rho, P1, P2, mu11, mu20, mu02,  # Moments
                       gpara1, gperp1, grot1,  # Friction coefficients
                       gpara2, gperp2, grot2,
                       vo, fs, ko, c, ks, beta, L1, L2,  # Other constants
                       fast=None):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param gpara1: Parallel drag coefficient of rod1
    @param gperp1: Perpendicular drag coefficient of rod1
    @param grot1: Rotational drag coefficient of rod1
    @param gpara2: Parallel drag coefficient of rod1
    @param gperp2: Perpendicular drag coefficient of rod1
    @param grot2: Rotational drag coefficient of rod1
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param fast: Flag on whether or not to use fast solving techniques
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    orient_change_arr = np.zeros(6)
    # Define useful parameters for functions
    a1 = np.dot(r12, u1)
    a2 = np.dot(r12, u2)
    b = np.dot(u1, u2)
    # Get average force of crosslinkers on rod2
    F12 = avg_force_zrl(r12, u1, u2, rho, P1, P2, ks)
    # Evolution of rod positions
    dr1 = dr_dt_zrl(-1. * F12, u1, gpara1, gperp1)
    dr2 = dr_dt_zrl(F12, u2, gpara2, gperp2)
    # Evolution of orientation vectors
    # du1 = du1_dt_zrl(r12, u1, u2, P1, mu11, a1, b, ks, grot1)
    # du2 = du2_dt_zrl(r12, u1, u2, P2, mu11, a2, b, ks, grot2)
    # Evolution of zeroth moment
    drho = drho_dt_zrl(rho, rsqr, a1, a2, b,
                       vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evoultion of first moments
    dP1 = dP1_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    dP2 = dP2_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, fast)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu20 = dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dmu02 = dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, fast)
    dsol = np.concatenate(
        (dr1, dr2, orient_change_arr, [drho, dP1, dP2, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_stat(rho, P1, P2, mu11, mu20, mu02,  # Moments
                     rsqr, a1, a2, b, q, q10, q01, q11, q20, q02,  # Pre-computed values
                     vo, fs, ko, c, ks, beta, L1, L2):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

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
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    rod_change_arr = np.zeros(12)
    # Get average force of crosslinkers on rod2
    # F12 = avg_force_zrl(r12, u1, u2, rho, P1, P2, ks)
    # Evolution of zeroth moment
    drho = drho_dt_zrl(rho, rsqr, a1, a2, b,
                       vo, fs, ko, c, ks, beta, L1, L2, q)
    # Evoultion of first moments
    dP1 = dP1_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, q10)
    dP2 = dP2_dt_zrl(rho, P1, P2,
                     rsqr, a1, a2, b,
                     vo, fs, ko, c, ks, beta, L1, L2, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q11)
    dmu20 = dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q20)
    dmu02 = dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                         a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2, q02)
    dsol = np.concatenate(
        (rod_change_arr, [drho, dP1, dP2, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol
