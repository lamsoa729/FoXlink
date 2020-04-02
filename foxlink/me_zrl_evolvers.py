#!/usr/bin/env python

"""@package docstring
File: me_zrl_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
# from scipy.integrate import dblquad
from .me_helpers import rod_geom_derivs, convert_sol_to_geom
from .me_zrl_odes import (rod_geom_derivs_zrl, calc_moment_derivs_zrl,
                          calc_boundary_derivs_zrl)
from .me_zrl_helpers import (avg_force_zrl, avg_torque_zrl, fast_zrl_src_kl,
                             prep_zrl_bound_evolver, prep_zrl_evolver,
                             get_zrl_moments,
                             get_zrl_moments_and_boundary_terms, get_mu_kl_eff)
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
    L_i, L_j = (params['L_i'], params['L_j'])
    rod_diam = params['rod_diam']
    beta = params['beta']
    ks = params['ks']
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    steric_flag = params.get('steric_interactions', None)

    # (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)

    # Get average force of crosslinkers on rod_j
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    tau_i = avg_torque_zrl(r_ij, u_i, u_j, mu_kl[1], mu_kl[3], ks)
    tau_j = avg_torque_zrl(-1. * r_ij, u_j, u_i, mu_kl[2], mu_kl[3], ks)

    # Calculate wca forces and torques if steric forces are set
    torque_i_wca, torque_j_wca = (np.zeros(3), np.zeros(3))
    if params['steric_flag'] == 'wca':
        # Get WCA steric forces and add them to crosslink forces
        eps_scale = 1.
        f_ij_wca, torque_i_wca, torque_j_wca = calc_wca_force_torque(
            r_i, r_j, u_i, u_j, L_i, L_j, rod_diam, eps_scale / beta, fcut=1e10)
        f_ij += f_ij_wca
        tau_i += torque_i_wca
        tau_j += torque_j_wca
    elif params['steric_flag'] == 'constrained':
        u_m = params['constr_vec']  # Get min dist vector of carrier lines
        f_ij -= np.dot(f_ij, u_m) * u_m  # Min dist component from force
        tau_i = np.dot(tau_i, u_m) * u_m  # Torque only around min dist vector
        tau_j = np.dot(tau_j, u_m) * u_m  # Torque only around min dist vector

    # dr_i, dr_j, du_i, du_j = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j,
        # scalar_geom, mu_kl,
        # fric_coeff, ks)
    dr_i, dr_j, du_i, du_j = rod_geom_derivs(f_ij, tau_i, tau_j,
                                             u_i, u_j, fric_coeff)

    # Add WCA torques if they exist to filament orientations
    # du_i += np.cross(torque_i_wca, u_i) / fric_coeff[2]
    # du_j += np.cross(torque_j_wca, u_j) / fric_coeff[5]

    # Moment evolution
    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)

    # Evolution of boundary condtions
    # dB_terms = calc_boundary_derivs_zrl(B_terms, scalar_geom, Q_arr, params)
    dB_terms = np.zeros(8)

    dsol = np.concatenate((dr_i, dr_j, du_i, du_j, dmu_kl, dB_terms))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. '
            'Current derivatives', dsol)
    return dsol


def evolver_zrl_bvg(sol, fric_coeff, params):
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
    ks = params['ks']
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i

    # (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)

    # Get effective mu_kl to calculate forces and torques.
    # This will simulate walking off the end of rods
    mu_kl_eff = get_mu_kl_eff(mu_kl, params)

    # Get average force of crosslinkers on rod_j and both torques.
    f_ij = avg_force_zrl(r_ij, u_i, u_j,
                         mu_kl_eff[0], mu_kl_eff[1], mu_kl_eff[2], ks)
    tau_i = avg_torque_zrl(r_ij, u_i, u_j, mu_kl_eff[1], mu_kl_eff[3], ks)
    tau_j = avg_torque_zrl(-r_ij, u_j, u_i, mu_kl_eff[2], mu_kl_eff[3], ks)

    if params['steric_flag'] == 'wca':
        # Get WCA steric forces and add them to crosslink forces
        eps_scale = 1.
        f_ij_wca, torque_i_wca, torque_j_wca = calc_wca_force_torque(
            r_i, r_j, u_i, u_j,
            params['L_i'], params['L_j'], params['rod_diam'],
            eps_scale / params['beta'], fcut=1e10)
        f_ij += f_ij_wca
        tau_i += torque_i_wca
        tau_j += torque_j_wca
    elif params['steric_flag'] == 'constrained':
        u_m = params['constr_vec']  # Get min dist vector of carrier lines
        f_ij -= np.dot(f_ij, u_m) * u_m  # Min dist component from force
        tau_i = np.dot(tau_i, u_m) * u_m  # Torque only around min dist vector
        tau_j = np.dot(tau_j, u_m) * u_m  # Torque only around min dist vector

    # dr_i, dr_j, du_i, du_j = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j,
    #                                              scalar_geom, mu_kl_eff,
    #                                              fric_coeff, ks)
    dr_i, dr_j, du_i, du_j = rod_geom_derivs(f_ij, tau_i, tau_j,
                                             u_i, u_j, fric_coeff)

    # # Add WCA torques if they exist to filament orientations
    # du_i += np.cross(torque_i_wca, u_i) / fric_coeff[2]
    # du_j += np.cross(torque_j_wca, u_j) / fric_coeff[5]

    # Moment evolution
    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)

    # Evolution of boundary condtions
    # dB_terms = calc_boundary_derivs_zrl(B_terms, scalar_geom, Q_arr, params)
    dB_terms = np.zeros(8)

    dsol = np.concatenate((dr_i, dr_j, du_i, du_j, dmu_kl, dB_terms))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)
    return dsol


def evolver_zrl_stat(mu_kl, scalar_geom, q_arr, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param mu_kl: Zeroth motor moment
    @param scalar_geom: First motor moment of s1
    @param q_arr: First motor moment of s2
    @param params: Second motor moment of s1 and s2
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    rod_change_arr = np.zeros(12)
    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)
    dB_terms = np.zeros(8)

    dsol = np.concatenate((rod_change_arr, dmu_kl, dB_terms))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)
    return dsol
