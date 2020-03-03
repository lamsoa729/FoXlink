#!/usr/bin/env python

"""@package docstring
File: me_zrl_odes.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Class that contains the all ODEs relevant to solving the moment
expansion formalism of the Fokker-Planck equation for bound crosslinking motors.
"""
from scipy.integrate import dblquad
from numba import njit
from .me_zrl_helpers import (boltz_fact_zrl, weighted_boltz_fact_zrl,
                             fast_zrl_src_kl)


@njit
def dui_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i):
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
    @param grot_j: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector
    """
    return (ks / grot_i) * ((r_ij - (a_ij * u_i))
                            * mu10 + (u_j - (b * u_i)) * mu11)


################################
#  Moment evolution functions  #
################################


@njit
def dmu00_dt_zrl(mu00, a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa,
                 q00=0, B0_j=0., B0_i=0., B1_j=0., B1_i=0.):
    """!Calculate the time-derivative of the zeroth moment of the zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param ko: Turnover rate of motors
    @param q00: Binding source term (i.e. partition function)
    @return: Time derivative of the zeroth moment of motors

    """
    return ko * (q00 - mu00) + ((-vo + kappa * (hL_i - a_ji)) * B0_j
                                - (kappa * b * B1_j)
                                + (-vo + kappa * (hL_j - a_ij)) * B0_i
                                - (kappa * b * B1_i))


@njit
def dmu10_dt_zrl(mu00, mu10, mu01,
                 a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa,
                 q10=0, B0_j=0, B1_j=0, B1_i=0, B2_i=0):
    """!Calculate the time-derivative of the first moment(s1) of the zero rest
    length crosslinkers bound to rods.

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param a_ij: Dot product of u_i and r_ij
    @param b: Dot product of u_i and u_j
    @param ko: Turnover rate of motors
    @param vo: Velocity of motor when no force is applied
    @param q10: Binding source term of first moment
    @return: Time derivative of the first(s1) moment of motors

    """
    return ((ko * q10) + ((vo + kappa * a_ij) * mu00) - ((ko + kappa) * mu10)
            + (kappa * b * mu01)
            + hL_i * (kappa * (hL_i - a_ij) - vo) * B0_j
            - kappa * b * hL_i * B1_j
            + (kappa * (hL_j - a_ji) - vo) * B1_i - kappa * b * B2_i)


@njit
def dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02,
                 a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa,
                 q11=0, B1_j=0, B1_i=0, B2_j=0, B2_i=0):
    """!Calculate the time-derivative of the second moment(s1,s2) of zero rest
     length crosslinkers bound to rods.

     @param mu10: First motor moment of s1
     @param mu01: First motor moment of s2
     @param mu11: Second motor moment of s1 and s2
     @param mu20: Second motor moment of s1
     @param mu02: Second motor moment of s2
     @param a_ij: Dot product of u_i and r_ij
     @param a_ji: Dot product of u_j and r_ij
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
    return ((ko * q11) + (vo + kappa * a_ji) * mu10 + (vo + kappa * a_ij) * mu01
            - (ko + 2. * kappa) * mu11 + kappa * b * (mu20 + mu02)
            + hL_i * ((kappa * (hL_i - a_ij) - vo) * B1_j - b * kappa * B2_j)
            + hL_j * ((kappa * (hL_j - a_ji) - vo) * B1_i - b * kappa * B2_i))


@njit
def dmu20_dt_zrl(mu10, mu11, mu20, a_ij, b, ko, vo, kappa, q20=0, a_ji=0,
                 hL_i=0, hL_j=0, B0_j=0, B1_j=0, B2_i=0, B3_i=0):
    """!Calculate the time-derivative of the second moment(s1^2) of zero rest
    length crosslinkers bound to rods.

    @param mu10: First motor moment of s1
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param a_ij: Dot product of u_i and r_ij
    @param b: Dot product of u_i and u_j
    @param ko: Turnover rate of motors
    @param vo: Velocity of motor when no force is applied
    @param kappa: Characteristic walking rate
    @return: Time derivative of the second(s1^2) moment of motors

    """
    return ((ko * q20) + 2. * (vo + kappa * a_ij) * mu10
            + (2. * kappa * b * mu11) - ((ko + 2. * kappa) * mu20)
            + (hL_i**2) * (kappa * (hL_i - a_ij) - vo) * B0_j
            - kappa * b * (hL_i**2) * B1_j
            + (kappa * (hL_j - a_ji) - vo) * B2_i - kappa * b * B3_i)


@njit
def dBl_j_dt_zrl(l, Bl_prev_j, Bl_j, a_ij, a_ji, b, s_i, vo, ko, kappa,
                 Ql_j=0):
    """!TODO: Docstring for dBl_j_dt_zrl.

    @param l: TODO
    @param Bl_prev_j: TODO
    @param Bl_j: TODO
    @param s_i: TODO
    @param vo: TODO
    @param ko: TODO
    @param kappa: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @return: TODO

    """
    # return (ko * Ql_j + l * (vo + kappa * (a_ji + b * s_i)) * Bl_prev_j
    # - (ko + kappa * (l - 1.)) * Bl_j)
    return -ko * (Ql_j - Bl_j)
