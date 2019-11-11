#!/usr/bin/env python

"""@package docstring
File: ME_gen_me_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from scipy.integrate import quad, dblquad
import numpy as np


def prep_gen_stat_2ord_me_evolver(sol, ks, beta, L_i, L_j):
    """!TODO: Docstring for prep_gen_stat_me_evolver.

    @param arg1: TODO
    @return: TODO

    """
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    q, e = dblquad(boltz_fact, -.5 * L_i, .5 * L_i,
                   lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                   args=[rsqr, a_ij, a_ji, b, ks, beta])
    q10, e = dblquad(weighted_boltz_fact, -.5 * L_i, .5 * L_i,
                     lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                     args=[1, 0, rsqr, a_ij, a_ji, b, ks, beta],)
    q01, e = dblquad(weighted_boltz_fact, -.5 * L_i, .5 * L_i,
                     lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                     args=[0, 1, rsqr, a_ij, a_ji, b, ks, beta])
    q11, e = dblquad(weighted_boltz_fact, -.5 * L_i, .5 * L_i,
                     lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                     args=[1, 1, rsqr, a_ij, a_ji, b, ks, beta])
    q20, e = dblquad(weighted_boltz_fact, -.5 * L_i, .5 * L_i,
                     lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                     args=[2, 0, rsqr, a_ij, a_ji, b, ks, beta])
    q02, e = dblquad(weighted_boltz_fact, -.5 * L_i, .5 * L_i,
                     lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                     args=[0, 2, rsqr, a_ij, a_ji, b, ks, beta])
    return rsqr, a_ij, a_ji, b, q, q10, q01, q11, q20, q02


def me_evolver_gen_2ord(r_i, r_j, u_i, u_j,  # Vectors
                        mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
                        gpara_ij, gperp_i, grot_i,  # Friction coefficients
                        gpara_ji, gperp_j, grot_j,
                        vo, fs, ko, co, ks, ho, beta, L_i, L_j, fast=None):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (gen) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param r_i: Center of mass postion of rod_i
    @param r_j: Center of mass position of rod_j
    @param u_i: Orientation unit vector of rod_i
    @param u_j: Orientation unit vector of rod_j
    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s_i
    @param mu01: First motor moment of s_j
    @param mu11: Second motor moment of s_i and s_j
    @param mu20: Second motor moment of s_i
    @param mu02: Second motor moment of s_j
    @param gpara_ij: Parallel drag coefficient of rod_i
    @param gperp_i: Perpendicular drag coefficient of rod_i
    @param grot_i: Rotational drag coefficient of rod_i
    @param gpara_ji: Parallel drag coefficient of rod_i
    @param gperp_j: Perpendicular drag coefficient of rod_i
    @param grot_j: Rotational drag coefficient of rod_i
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod_i
    @param L_j: Length of rod_j
    @param fast: Flag on whether or not to use fast solving techniques
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1. * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)
    # Get average force of crosslinkers on rod_j
    F_ij = avg_force_gen(r_ij, u_i, u_j, mu00, mu10, mu01, ks)
    # Evolution of rod positions
    dr_i = dr_dt_gen(-1. * F_ij, u_i, gpara_i, gperp_i)
    dr_j = dr_dt_gen(F_ij, u_j, gpara_j, gperp_j)
    # Evolution of orientation vectors
    du_i = du_dt_gen(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i)
    du_j = du_dt_gen(-1. * r_ij, u_j, u_i, mu01, mu11, a_ji, b, ks, grot_j)
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_gen(mu00, rsqr, a_ij, a_ji, b,
                         vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    # Evoultion of first moments
    dmu10 = dmu10_dt_gen(mu00, mu10, mu01,
                         rsqr, a_ij, a_ji, b,
                         vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    dmu01 = dmu01_dt_gen(mu00, mu10, mu01,
                         rsqr, a_ij, a_ji, b,
                         vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    # Evolution of second moments
    dmu11 = dmu11_dt_gen(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                         a_ij, a_ji, b, vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    dmu20 = dmu20_dt_gen(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                         a_ij, a_ji, b, vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    dmu02 = dmu02_dt_gen(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
                         a_ij, a_ji, b, vo, fs, ko, co, ks, beta, L_i, L_j, fast)
    dsol = np.concatenate(
        (dr_i, dr_j, du_i, du_j, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol
