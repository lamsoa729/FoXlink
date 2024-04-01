#!/usr/bin/env python

"""@package docstring
File: me_zrl_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np

# from scipy.integrate import dblquad
from .me_helpers import convert_nfil_sol_to_geom, dr_dt, du_dt
from .me_zrl_helpers import (
    pair_index,
    avg_force_zrl,
    avg_torque_zrl,
    prep_zrl_nfil_evolver,
    get_zrl_xl_moments_for_ij,
)

from .me_zrl_xl_odes import calc_zrl_xl_moment_derivs


def me_evolver_nfil_crosslink(sol, fric_coeff_arr, params):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable.
    Currently only works for static crosslinkers.

    @param sol: Solution vector to solve_ivp, first
    @param fric_coeff: friction coefficients of rod
    @param params: Constant parameters of the simulation
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """

    ks = params["ks"]
    ko = params["ko"]
    n_fils = len(fric_coeff_arr)

    derivs = np.zeros((n_fils * 7 + n_fils * (n_fils - 1) * 2 + 1))

    force_arr = np.zeros((n_fils, 3))
    torque_arr = np.zeros((n_fils, 3))

    # Loop over pairs of rods and calculate forces and torques
    for i in range(n_fils):
        # Get geometry for i
        r_i, u_i, L_i = convert_nfil_sol_to_geom(sol, i)
        for j in range(n_fils):
            # Get geometry for j
            r_j, u_j, L_j = convert_nfil_sol_to_geom(sol, j)
            r_ij = r_j - r_i

            (scalar_geom, q_arr) = prep_zrl_nfil_evolver(
                r_i, u_i, L_i, r_j, u_j, L_j, params
            )

            # mu_kl = [mu^{00}, mu^{01}, mu^{10}, mu^{11}]
            n_unbound, mu_kl = get_zrl_xl_moments_for_ij(sol, i, j, n_fils)

            # Get average force of crosslinkers on rod_j
            force_arr[i] += avg_force_zrl(
                r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks
            )
            force_arr[j] -= force_arr[i]
            torque_arr[i] += avg_torque_zrl(r_ij, u_i, u_j, mu_kl[1], mu_kl[3], ks)
            torque_arr[j] += avg_torque_zrl(
                -1.0 * r_ij, u_j, u_i, mu_kl[2], mu_kl[3], ks
            )

            # Moment evolution
            dn_unbound, dmu_kl = calc_zrl_xl_moment_derivs(n_unbound, mu_kl, q_arr, ko)

            derivs[-1] += dn_unbound
            ij = n_fils * 7 + pair_index(i, j, n_fils)
            derivs[ij : ij + 4] = dmu_kl

        # TODO: add in boundary forces eventually

        # Now that we have collected all the forces and torques on rod i, calculate positional derivatives
        i_start = i * 7
        drag_para, drag_perp, drag_rot = fric_coeff_arr[i]
        derivs[i_start : i_start + 3] = dr_dt(force_arr[i], u_i, drag_para, drag_perp)
        derivs[i_start + 3 : i_start + 6] = du_dt(torque_arr[i], u_i, drag_rot)

    # Check to make sure all values are finite
    if not np.all(np.isfinite(derivs)):
        raise RuntimeError(
            "Infinity or NaN thrown in ODE solver derivatives. " "Current derivatives",
            derivs,
        )
    return derivs
