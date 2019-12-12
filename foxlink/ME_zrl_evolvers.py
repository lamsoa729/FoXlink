#!/usr/bin/env python

"""@package docstring
File: ME_zrl_ODEs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
# from scipy.integrate import dblquad
from .ME_helpers import dr_dt, convert_sol_to_geom
from .ME_zrl_ODEs import (dui_dt_zrl,
                          dmu00_dt_zrl, dmu10_dt_zrl, dmu01_dt_zrl,
                          dmu11_dt_zrl, dmu20_dt_zrl, dmu02_dt_zrl)
from .ME_zrl_helpers import (avg_force_zrl, boltz_fact_zrl,
                             weighted_boltz_fact_zrl, fast_zrl_src_kl)


def get_zrl_moments(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    return sol[12:18].tolist()


def prep_zrl_evolver(sol, c, ks, beta, L_i, L_j):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param arg1: TODO
    @return: TODO

    """
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1. * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    q00 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=0)
    # q00, e = dblquad(boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    #                  lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    #                  args=[rsqr, a_ij, a_ji, b, ks, beta])
    q10 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji,
                              a_ij, b, ks, beta, k=0, l=1)
    # q10, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    # lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    # args=[1, 0, rsqr, a_ij, a_ji, b, ks, beta],)
    q01 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=1)
    # q01, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    # lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    # args=[0, 1, rsqr, a_ij, a_ji, b, ks, beta])
    q11 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=1, l=1)
    # q11, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    # lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    # args=[1, 1, rsqr, a_ij, a_ji, b, ks, beta])
    q20 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji,
                              a_ij, b, ks, beta, k=0, l=2)
    # q20, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    # lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    # args=[2, 0, rsqr, a_ij, a_ji, b, ks, beta])
    q02 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij,
                              a_ji, b, ks, beta, k=0, l=2)
    # q02, e = dblquad(weighted_boltz_fact_zrl, -.5 * L_i, .5 * L_i,
    # lambda s2: -.5 * L_j, lambda s2: .5 * L_j,
    # args=[0, 2, rsqr, a_ij, a_ji, b, ks, beta])
    return rsqr, a_ij, a_ji, b, q00, q10, q01, q11, q20, q02


def evolver_zrl(sol,
                gpara_i, gperp_i, grot_i,  # Friction coefficients
                gpara_j, gperp_j, grot_j,
                vo, fs, ko, c, ks, beta, L_i, L_j):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Solution vector to solve_ivp
    @param gpara_i: Parallel drag coefficient of rod1
    @param gperp_i: Perpendicular drag coefficient of rod1
    @param grot_i: Rotational drag coefficient of rod1
    @param gpara_j: Parallel drag coefficient of rod1
    @param gperp_j: Perpendicular drag coefficient of rod1
    @param grot_j: Rotational drag coefficient of rod1
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param c: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod1
    @param L_j: Length of rod2
    @param fast: Flag on whether or not to use fast solving techniques
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(sol, c, ks, beta, L_i, L_j)
    mu00, mu10, mu01, mu11, mu20, mu02 = get_zrl_moments(sol)

    # Get average force of crosslinkers on rod2
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu00, mu10, mu01, ks)
    # Evolution of rod positions
    dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
    dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
    # Evolution of orientation vectors
    du_i = dui_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i)
    du_j = dui_dt_zrl(-1. * r_ij, u_j, u_i, mu01, mu11, a_ji, b, ks, grot_j)

    # Characteristic walking rate
    kappa = vo * ks / fs
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_zrl(mu00, ko, q00)
    # Evoultion of first moments
    # TODO Double check this for accuracy
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, b, ko, vo, kappa, q10)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, b, ko, vo, kappa, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         ko, vo, kappa, q11)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, b, ko, vo, kappa, q20)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, b, ko, vo, kappa, q02)
    dsol = np.concatenate(
        (dr_i, dr_j, du_i, du_j, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_stat(mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
                     a_ij, a_ji, b,
                     q00, q10, q01, q11, q20, q02,  # Pre-computed values
                     vo, fs, ko, ks):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param mu00: Zeroth motor moment
    @param mu10: First motor moment of s1
    @param mu01: First motor moment of s2
    @param mu11: Second motor moment of s1 and s2
    @param mu20: Second motor moment of s1
    @param mu02: Second motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM (r_ij)
    @param a_ij: Dot product of u_i and r_ij
    @param a_ji: Dot product of u_j and r_ij
    @param b: Dot product of u_i and u_j
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    rod_change_arr = np.zeros(12)
    # Characteristic walking rate
    kappa = vo * ks / fs
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_zrl(mu00, ko, q00)
    # Evoultion of first moments
    # TODO Double check this for accuracy
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, b, ko, vo, kappa, q10)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, b, ko, vo, kappa, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         ko, vo, kappa, q11)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, b, ko, vo, kappa, q20)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, b, ko, vo, kappa, q02)
    dsol = np.concatenate(
        (rod_change_arr, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


# def evolver_zrl_ang(sol, mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
#                     grot_i, grot_j,  # Friction coefficients
#                     vo, fs, ko, c, ks, beta, L_i, L_j):
#     """!Calculate all time derivatives necessary to solve the moment expansion
#     evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
# bound to moving rods. d<var> is the time derivative of corresponding
# variable

#     @param u_i: Orientation unit vector of rod1
#     @param u_j: Orientation unit vector of rod2
#     @param mu00: Zeroth motor moment
#     @param mu10: First motor moment of s1
#     @param mu01: First motor moment of s2
#     @param mu11: Second motor moment of s1 and s2
#     @param mu20: Second motor moment of s1
#     @param mu02: Second motor moment of s2
#     @param gpara_i: Parallel drag coefficient of rod1
#     @param gperp_i: Perpendicular drag coefficient of rod1
#     @param grot_i: Rotational drag coefficient of rod1
#     @param gpara_j: Parallel drag coefficient of rod1
#     @param gperp_j: Perpendicular drag coefficient of rod1
#     @param grot_j: Rotational drag coefficient of rod1
#     @param vo: Velocity of motor when no force is applied
#     @param fs: Stall force of motor ends
#     @param ko: Turnover rate of motors
#     @param c: Effective concentration of motors in solution
#     @param ks: Motor spring constant
#     @param beta: 1/(Boltzmann's constant * Temperature)
#     @param L_i: Length of rod1
#     @param L_j: Length of rod2
#     @param fast: Flag on whether or not to use fast solving techniques
#     @return: Time-derivatives of all time varying quantities in a flattened
#              array
#     """
#     rod_change_arr = np.zeros(6)
#     # Define useful parameters for functions
#     rsqr = np.dot(r_ij, r_ij)
#     a_ij = np.dot(r_ij, u_i)
#     a_ji = -1. * np.dot(r_ij, u_j)
#     b = np.dot(u_i, u_j)
#     # Get average force of crosslinkers on rod2
#     # f_ij = avg_force_zrl(r_ij, u_i, u_j, mu00, mu10, mu01, ks)
#     # Evolution of rod positions
#     # dr_i = dr_dt_zrl(-1. * f_ij, u_i, gpara_i, gperp_i)
#     # dr_j = dr_dt_zrl(f_ij, u_j, gpara_j, gperp_j)
#     # Evolution of orientation vectors
#     du1 = du1_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i)
#     du2 = du2_dt_zrl(r_ij, u_i, u_j, mu01, mu11, a_ji, b, ks, grot_j)
#     # Evolution of zeroth moment
#     dmu00 = dmu00_dt_zrl(mu00, rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     # Evoultion of first moments
#     dmu10 = dmu10_dt_zrl(mu00, mu10, mu01,
#                          rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu01 = dmu01_dt_zrl(mu00, mu10, mu01,
#                          rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     # Evolution of second moments
#     dmu11 = dmu11_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu20 = dmu20_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu02 = dmu02_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dsol = np.concatenate(
#         (rod_change_arr, du1, du2, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
#     # Check to make sure all values are finite
#     if not np.all(np.isfinite(dsol)):
#         raise RuntimeError(
#             'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

#     return dsol


# def evolver_zrl_orient(r_i, r_j, u_i, u_j,  # Vectors
#                        mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
#                        gpara_i, gperp_i, grot_i,  # Friction coefficients
#                        gpara_j, gperp_j, grot_j,
#                        vo, fs, ko, c, ks, beta, L_i, L_j,  # Other constants
#                        fast='fast'):
#     """!Calculate all time derivatives necessary to solve the moment expansion
#     evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
# bound to moving rods. d<var> is the time derivative of corresponding
# variable

#     @param u_i: Orientation unit vector of rod1
#     @param u_j: Orientation unit vector of rod2
#     @param mu00: Zeroth motor moment
#     @param mu10: First motor moment of s1
#     @param mu01: First motor moment of s2
#     @param mu11: Second motor moment of s1 and s2
#     @param mu20: Second motor moment of s1
#     @param mu02: Second motor moment of s2
#     @param gpara_i: Parallel drag coefficient of rod1
#     @param gperp_i: Perpendicular drag coefficient of rod1
#     @param grot_i: Rotational drag coefficient of rod1
#     @param gpara_j: Parallel drag coefficient of rod1
#     @param gperp_j: Perpendicular drag coefficient of rod1
#     @param grot_j: Rotational drag coefficient of rod1
#     @param vo: Velocity of motor when no force is applied
#     @param fs: Stall force of motor ends
#     @param ko: Turnover rate of motors
#     @param c: Effective concentration of motors in solution
#     @param ks: Motor spring constant
#     @param beta: 1/(Boltzmann's constant * Temperature)
#     @param L_i: Length of rod1
#     @param L_j: Length of rod2
#     @param fast: Flag on whether or not to use fast solving techniques
#     @return: Time-derivatives of all time varying quantities in a flattened
#              array
#     """
#     r_ij = r_j - r_i
#     rsqr = np.dot(r_ij, r_ij)
#     a_ij = np.dot(r_ij, u_i)
#     a_ji = -1. * np.dot(r_ij, u_j)
#     b = np.dot(u_i, u_j)
#     orient_change_arr = np.zeros(6)
#     # Define useful parameters for functions
#     a_ij = np.dot(r_ij, u_i)
#     a_ji = np.dot(r_ij, u_j)
#     b = np.dot(u_i, u_j)
#     # Get average force of crosslinkers on rod2
#     f_ij = avg_force_zrl(r_ij, u_i, u_j, mu00, mu10, mu01, ks)
#     # Evolution of rod positions
#     dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
#     dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
#     # Evolution of orientation vectors
#     # du1 = du1_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i)
#     # du2 = du2_dt_zrl(r_ij, u_i, u_j, mu01, mu11, a_ji, b, ks, grot_j)
#     # Evolution of zeroth moment
#     dmu00 = dmu00_dt_zrl(mu00, rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     # Evoultion of first moments
#     dmu10 = dmu10_dt_zrl(mu00, mu10, mu01,
#                          rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu01 = dmu01_dt_zrl(mu00, mu10, mu01,
#                          rsqr, a_ij, a_ji, b,
#                          vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     # Evolution of second moments
#     dmu11 = dmu11_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu20 = dmu20_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dmu02 = dmu02_dt_zrl(mu00, mu10, mu01, mu11, mu20, mu02, rsqr,
#                          a_ij, a_ji, b, vo, fs, ko, c, ks, beta, L_i, L_j, fast)
#     dsol = np.concatenate(
#         (dr_i, dr_j, orient_change_arr, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
#     # Check to make sure all values are finite
#     if not np.all(np.isfinite(dsol)):
#         raise RuntimeError(
#             'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

#     return dsol
