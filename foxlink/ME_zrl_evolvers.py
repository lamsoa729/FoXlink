#!/usr/bin/env python
import numpy as np
from .ME_zrl_ODEs import (dr_dt_zrl, du1_dt_zrl, du2_dt_zrl,
                          drho_dt_zrl, dP1_dt_zrl, dP2_dt_zrl,
                          dmu11_dt_zrl, dmu20_dt_zrl, dmu02_dt_zrl)
from .ME_zrl_helpers import avg_force_zrl

"""@package docstring
File: ME_zrl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


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
