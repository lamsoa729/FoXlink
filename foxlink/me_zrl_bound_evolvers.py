#!/usr/bin/env python

"""@package docstring
File: me_zrl_bound_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
# from scipy.integrate import dblquad
from .me_helpers import dr_dt, convert_sol_to_geom
from .me_zrl_odes import (rod_geom_derivs_zrl, calc_moment_derivs_zrl,
                          calc_moment_derivs_zrl_B_terms,
                          calc_boundary_derivs_zrl)
from .me_zrl_helpers import (avg_force_zrl,
                             prep_zrl_bound_evolver,
                             get_zrl_moments_and_boundary_terms)
from .rod_steric_forces import calc_wca_force_torque
from .me_zrl_evolvers import prep_zrl_evolver


def evolver_zrl_bound(sol, fric_coeff, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
bound to moving rods. d<var> is the time derivative of corresponding
variable

    @param sol: Solution vector to solve_ivp
    @param fric_coeff: friction coefficients of rod
    @param params: Constant parameters of the simulation
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Define useful parameters for functions
    hL_i, hL_j = (.5 * params['L_i'], .5 * params['L_j'])
    ks = params['ks']
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i

    (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)
    if mu_kl[0] < 0.:
        mu_kl[0] = 0.
    if mu_kl[4] < 0.:
        mu_kl[4] = 0.
    if mu_kl[5] < 0.:
        mu_kl[5] = 0.

    # Get average force of crosslinkers on rod2
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    # Evolution of rod positions
    dgeom = rod_geom_derivs_zrl(f_ij, r_ij, u_i, u_j, scalar_geom,
                                mu_kl, fric_coeff, ks)

    # Evolution of moments
    dmu_kl = calc_moment_derivs_zrl_B_terms(mu_kl, scalar_geom,
                                            q_arr, B_terms, params)

    # Evolution of boundary condtions
    dB_terms = calc_boundary_derivs_zrl(B_terms, scalar_geom, Q_arr, params)
    dsol = np.concatenate(dgeom, dmu_kl, dB_terms)
    return dsol

##########################################
