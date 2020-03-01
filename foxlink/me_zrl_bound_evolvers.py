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
from .me_zrl_odes import (dui_dt_zrl, dmu00_dt_zrl, dmu10_dt_zrl,
                          dmu11_dt_zrl, dmu20_dt_zrl)
from .me_zrl_helpers import (avg_force_zrl, fast_zrl_src_integrand_l0,
                             fast_zrl_src_integrand_l1,
                             fast_zrl_src_integrand_l2)
from .rod_steric_forces import calc_wca_force_torque
from .me_zrl_evolvers import prep_zrl_evolver


def get_zrl_moments_and_boundary_terms(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    return sol[12:24].tolist()


def get_Qj_params(s_i, L_j, a_ji, b, ks, beta):
    hL_j = .5 * L_j
    sigma = np.sqrt(2. / (ks * beta))
    A_j = -1. * (a_ji + (b * s_i))
    return hL_j, sigma, A_j


def prep_zrl_boundary_evolver(sol, c, ks, beta, L_i, L_j):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param arg1: TODO
    @return: TODO

    """
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(sol, c, ks,
                                                      beta, L_i, L_j)

    hL_j, sigma, A_j = get_Qj_params(.5 * L_i, L_j, a_ji, b, ks, beta)
    hL_i, sigma, A_i = get_Qj_params(hL_j, L_i, a_ij, b, ks, beta)
    Q_j0 = c * fast_zrl_src_integrand_l0(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q_i0 = c * fast_zrl_src_integrand_l0(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q_j1 = c * fast_zrl_src_integrand_l1(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q_i1 = c * fast_zrl_src_integrand_l1(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q_j2 = c * fast_zrl_src_integrand_l2(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q_i2 = c * fast_zrl_src_integrand_l2(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    return (rsqr, a_ij, a_ji, b,
            q00, q10, q01, q11, q20, q02,
            Q_j0, Q_i0, Q_j1, Q_i1, Q_j2, Q_i2)


def evolver_zrl_bound(sol,
                      gpara_i, gperp_i, grot_i,  # Friction coefficients
                      gpara_j, gperp_j, grot_j,
                      vo, fs, ko, c, ks, beta, L_i, L_j):  # Other constants
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (zrl) crosslinkers
bound to moving rods. d<var> is the time derivative of corresponding
variable

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
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02,
     Q_j0, Q_i0, Q_j1, Q_i1, Q_j2, Q_i2) = prep_zrl_boundary_evolver(
        sol, c, ks, beta, L_i, L_j)
    (mu00, mu10, mu01, mu11, mu20, mu02,
     Q_j0, Q_i0, Q_j1, Q_i1, Q_j2, Q_i2) = get_zrl_moments_and_boundary_terms(sol)

    dsol = 0
    return dsol

##########################################
