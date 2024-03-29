#!/usr/bin/env python

"""@package docstring
File: me_zrl_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np

# from scipy.integrate import dblquad
from .me_helpers import rod_geom_derivs, convert_nfil_sol_to_geom
from .me_zrl_odes import (
    rod_geom_derivs_zrl,
    calc_moment_derivs_zrl,
    calc_zrl_xl_moment_derivs,  # TODO make this
)
from .me_zrl_helpers import (
    pair_index,
    avg_force_zrl,
    avg_torque_zrl,
    prep_zrl_nfil_evolver,
    get_zrl_xl_moments_for_rods_ij,
)


def me_evolver_nfil_crosslink(sol, fric_coeff, params):
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
    # TODO NEXT This is the function that needs major overhauling

    ks = params["ks"]
    n_fils = params["n_filaments"]

    derivs = np.zeros((n_fils * 7 + n_fils * (n_fils - 1) * 2 + 1))

    # Loop over pairs of rods and calculate forces and torques
    for i in range(n_fils):
        for j in range(n_fils):
            # Get geometry for the pair of rods
            r_i, u_i, L_i = convert_nfil_sol_to_geom(sol, i)
            r_j, u_j, L_j = convert_nfil_sol_to_geom(sol, j)
            r_ij = r_j - r_i

            # (scalar_geom, q_arr, Q_arr) = prep_zrl_bound_evolver(sol, params)
            (scalar_geom, q_arr) = prep_zrl_nfil_evolver(
                r_i, u_i, L_i, r_j, u_j, L_j, params
            )
            # mu_kl = [mu^{00}, mu^{01}, mu^{10}, mu^{11}]
            n_unbound, mu_kl = get_zrl_xl_moments_for_rods_ij(sol, i, j, n_fils)

            # Get average force of crosslinkers on rod_j
            f_ij = avg_force_zrl(r_ij, u_i, u_j, mu_kl[0], mu_kl[1], mu_kl[2], ks)
            tau_i = avg_torque_zrl(r_ij, u_i, u_j, mu_kl[1], mu_kl[3], ks)
            tau_j = avg_torque_zrl(-1.0 * r_ij, u_j, u_i, mu_kl[2], mu_kl[3], ks)

            # Geometry evolution
            # TODO fix friction coefficients
            dr_i, dr_j, du_i, du_j = rod_geom_derivs(
                f_ij, tau_i, tau_j, u_i, u_j, fric_coeff[i] + fric_coeff[j]
            )

            # Moment evolution
            dn_unbound, dmu_kl = calc_zrl_xl_moment_derivs(
                n_unbound, mu_kl, scalar_geom, q_arr, params
            )

            # Fill in derivatives
            i_start = i * 7
            derivs[i_start : i_start + 3] = dr_i
            derivs[i_start + 3 : i_start + 6] = du_i

            j_start = j * 7
            derivs[j_start : j_start + 3] = dr_j
            derivs[j_start + 3 : j_start + 6] = du_j

            ij = n_fils * 7 + pair_index(i, j, n_fils)
            derivs[ij : ij + 4] = dmu_kl

            derivs[-1] += dn_unbound

    dsol = np.concatenate((dr_i, dr_j, du_i, du_j, dmu_kl))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            "Infinity or NaN thrown in ODE solver derivatives. " "Current derivatives",
            dsol,
        )
    return dsol
