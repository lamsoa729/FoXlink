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
from .me_zrl_odes import (
    rod_geom_derivs_zrl,
    calc_moment_derivs_zrl,
    calc_boundary_derivs_zrl,
)
from .me_zrl_helpers import (
    avg_force_zrl,
    avg_torque_zrl,
    fast_zrl_src_kl,
    prep_zrl_bound_evolver,
    prep_zrl_evolver,
    get_zrl_moments,
    get_zrl_moments_and_boundary_terms,
    get_mu_kl_eff,
)


def me_evolver_nfil_crosslink(sol, fric_coeff, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Solution vector to solve_ivp
    @param fric_coeff: friction coefficients of rod
    @param params: Constant parameters of the simulation
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # TODO NEXT This is the function that needs major overhauling

    ks = params["ks"]

    # Loop over pairs of rods and calculate forces and torques
    # Define useful parameters for functions
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i

    # (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    (mu_kl, B_terms) = get_zrl_moments_and_boundary_terms(sol)

    # Get average force of crosslinkers on rod_j
    f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
    tau_i = avg_torque_zrl(r_ij, u_i, u_j, mu_kl[1], mu_kl[3], ks)
    tau_j = avg_torque_zrl(-1.0 * r_ij, u_j, u_i, mu_kl[2], mu_kl[3], ks)

    dr_i, dr_j, du_i, du_j = rod_geom_derivs(f_ij, tau_i, tau_j, u_i, u_j, fric_coeff)

    # Moment evolution
    dmu_kl = calc_moment_derivs_zrl(mu_kl, scalar_geom, q_arr, params)

    # Evolution of boundary condtions
    # dB_terms = calc_boundary_derivs_zrl(B_terms, scalar_geom, Q_arr, params)
    dB_terms = np.zeros(8)

    dsol = np.concatenate((dr_i, dr_j, du_i, du_j, dmu_kl, dB_terms))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            "Infinity or NaN thrown in ODE solver derivatives. " "Current derivatives",
            dsol,
        )
    return dsol
