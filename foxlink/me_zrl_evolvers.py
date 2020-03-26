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
                          dmu11_dt_zrl, dmu20_dt_zrl, dBl_j_dt_zrl,
                          rod_geom_derivs_zrl, calc_moment_derivs_zrl,
                          calc_boundary_derivs_zrl)
from .me_zrl_helpers import (avg_force_zrl, fast_zrl_src_kl,
                             prep_zrl_bound_evolver, prep_zrl_evolver,
                             get_zrl_moments,
                             get_zrl_moments_and_boundary_terms)
from .rod_steric_forces import calc_wca_force_torque


def evolver_zrl(sol, fric_coeff, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Solution vector to solve_ivp
    @param fric_coeff: friction coefficients of rod
    @param params: Constant parameters of the simulation
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    hL_i, hL_j = (.5 * params['L1'], .5 * params['L2'])
    ks = params['ks']
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i

    (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)

    # Get average force of crosslinkers on rod2
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    dgeom = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j, scalar_geom,
                                mu_kl, fric_coeff, ks)

    # Moment evolution
    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)

    # Evolution of boundary condtions
    dB_terms = calc_boundary_derivs_zrl(B_terms, scalar_geom, Q_arr, params)

    dsol = np.concatenate(dgeom, dmu_kl, dB_terms)
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)
    return dsol


def evolver_zrl_wca(sol, fric_coeff, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Solution vector to solve_ivp
    @param fric_coeff:
    @param params:
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    mu_kl = get_zrl_moments(sol)

    L_i, L_j = params['L1'], params['L2']
    ks = params['ks']
    beta = params['beta']
    rod_diameter = params['rod_diameter']

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

    # Add WCA torque to ith filament
    dgeom[2] += np.cross(torque_i_wca, u_i)
    dgeom[3] += np.cross(torque_j_wca, u_j)

    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)

    dsol = np.concatenate(dgeom, dmu_kl, [0] * 8)
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def evolver_zrl_stat(mu_kl, scalar_geom, q_arr, params):
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
