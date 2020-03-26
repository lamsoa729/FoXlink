#!/usr/bin/env python

"""@package docstring
File: me_zrl_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
# from scipy.integrate import dblquad
from .me_helpers import dr_dt, convert_sol_to_geom
from .me_zrl_odes import (dui_dt_zrl, dmu00_dt_zrl, dmu10_dt_zrl,
                          dmu11_dt_zrl, dmu20_dt_zrl, dBl_j_dt_zrl)
from .me_zrl_helpers import (avg_force_zrl, fast_zrl_src_kl,
                             prep_zrl_bound_evolver, prep_zrl_evolver,
                             get_zrl_moments,
                             get_zrl_moments_and_boundary_terms,
                             rod_geom_derivs_zrl)
from .rod_steric_forces import calc_wca_force_torque


def evolver_zrl(sol, fric_coeff,
                vo, fs, ko, c, ks, beta, L_i, L_j):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Solution vector to solve_ivp
    @param fric_coeff: friction coefficients of rod
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
    hL_i, hL_j = (.5 * L_i, .5 * L_j)
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i

    # (rsqr, a_ij, a_ji, b,
    # q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(sol, c, ks, beta, L_i,
    # L_j)
    # mu00, mu10, mu01, mu11, mu20, mu02 = get_zrl_moments(sol)
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02,
     Q0_j, Q0_i, Q1_j, Q1_i,
     Q2_j, Q2_i, Q3_j, Q3_i) = prep_zrl_bound_evolver(sol, c, ks, beta, L_i, L_j)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)

    scalar_geom = (rsqr, a_ij, a_ji, b)

    # Get average force of crosslinkers on rod2
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    dgeom = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j, scalar_geom,
                                mu_kl, fric_coeff, ks)

    # Evolution of rod positions
    # dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
    # dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
    # # Evolution of orientation vectors
    # du_i = dui_dt_zrl(r_ij, u_i, u_j, mu10, mu11, a_ij, b, ks, grot_i)
    # du_j = dui_dt_zrl(-1. * r_ij, u_j, u_i, mu01, mu11, a_ji, b, ks, grot_j)

    # Characteristic walking rate
    kappa = vo * ks / fs
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_zrl(mu00, a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa, q00)
    # Evoultion of first moments
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q10)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         hL_j, hL_i, ko, vo, kappa, q11)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q20)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q02)
    # Evolution of boundary condtions
    dB0_j = dBl_j_dt_zrl(0., 0., B0_j, a_ij, a_ji, b, hL_i, vo, ko, kappa,
                         Q0_j)
    dB0_i = dBl_j_dt_zrl(0., 0., B0_i, a_ji, a_ij, b, hL_j, vo, ko, kappa,
                         Q0_i)
    dB1_j = dBl_j_dt_zrl(1., B0_j, B1_j, a_ij, a_ji, b, hL_i, vo, ko, kappa,
                         Q1_j)
    dB1_i = dBl_j_dt_zrl(1., B0_i, B1_i, a_ji, a_ij, b, hL_j, vo, ko, kappa,
                         Q1_i)
    dB2_j = dBl_j_dt_zrl(2., B1_j, B2_j, a_ij, a_ji, b, hL_i, vo, ko, kappa,
                         Q2_j)
    dB2_i = dBl_j_dt_zrl(2., B1_i, B2_i, a_ji, a_ij, b, hL_j, vo, ko, kappa,
                         Q2_i)
    dB3_j = dBl_j_dt_zrl(3., B2_j, B3_j, a_ij, a_ji, b, hL_i, vo, ko, kappa,
                         Q3_j)
    dB3_i = dBl_j_dt_zrl(3., B2_i, B3_i, a_ji, a_ij, b, hL_j, vo, ko, kappa,
                         Q3_i)
    dsol = np.concatenate((dgeom,
                           [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02,
                            dB0_j, dB0_i, dB1_j, dB1_i, dB2_j, dB2_i, dB3_j, dB3_i]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)
    return dsol


def evolver_zrl_wca(sol, fric_coeff,
                    vo, fs, ko, c, ks, beta, L_i, L_j, rod_diameter):  # Other constants
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
    @param L_i: Length of rod_i
    @param L_j: Length of rod_j
    @param rod_diameter: Diameter of rods
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    hL_i, hL_j = (.5 * L_i, .5 * L_j)
    r_ij = r_j - r_i
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(sol, c, ks, beta, L_i, L_j)
    mu_kl = get_zrl_moments(sol)
    scalar_geom = (rsqr, a_ij, a_ji, b)

    # Get average force of crosslinkers on rod2
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    # Get WCA steric forces and add them to crosslink forces
    eps_scale = 1.
    f_ij_wca, torque_i_wca, torque_j_wca = calc_wca_force_torque(
        r_i, r_j, u_i, u_j, L_i, L_j, rod_diameter, eps_scale / beta, fcut=1e22)

    f_ij += f_ij_wca

    # Evolution of rod positions
    dgeom = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j, scalar_geom,
                                mu_kl, fric_coeff, ks)

    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr,
                                    hL_i, hL_j, ko, vo, fs, ks)
    # Characteristic walking rate
    kappa = vo * ks / fs
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_zrl(mu00, a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa, q00)
    # Evoultion of first moments
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q10)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         hL_j, hL_i, ko, vo, kappa, q11)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q20)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q02)
    dsol = np.concatenate((dr_i, dr_j, du_i, du_j,
                           [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02], [0] * 8))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_stat(mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
                     a_ij, a_ji, b, L_i, L_j,
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
    hL_i, hL_j = (.5 * L_i, .5 * L_j)
    # Define useful parameters for functions
    rod_change_arr = np.zeros(12)
    # Characteristic walking rate
    kappa = vo * ks / fs
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_zrl(mu00, a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa, q00)
    # Evoultion of first moments
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q10)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         hL_j, hL_i, ko, vo, kappa, q11)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q20)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q02)
    dsol = np.concatenate(
        (rod_change_arr, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02], [0] * 8))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)
    return dsol
