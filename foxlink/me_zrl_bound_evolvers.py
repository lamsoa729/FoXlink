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
from .me_zrl_helpers import (avg_force_zrl, semi_anti_deriv_boltz_0,
                             semi_anti_deriv_boltz_1, semi_anti_deriv_boltz_2)
from .rod_steric_forces import calc_wca_force_torque
from .me_zrl_evolvers import prep_zrl_evolver


def get_zrl_moments_and_boundary_terms(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    # return sol[12:18].tolist()
    pass


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
    Q_j0 = c * (semi_anti_deriv_boltz_0(hL_j, sigma, A_j) -
                semi_anti_deriv_boltz_0(hL_j, sigma, A_j))
    Q_i0 = c * (semi_anti_deriv_boltz_0(hL_i, sigma, A_i) -
                semi_anti_deriv_boltz_0(hL_i, sigma, A_i))

    Q_j1 = c * (semi_anti_deriv_boltz_1(hL_j, sigma, A_j) -
                semi_anti_deriv_boltz_1(hL_j, sigma, A_j))
    Q_i1 = c * (semi_anti_deriv_boltz_1(hL_i, sigma, A_i) -
                semi_anti_deriv_boltz_1(hL_i, sigma, A_i))

    Q_j2 = c * (semi_anti_deriv_boltz_2(hL_j, sigma, A_j) -
                semi_anti_deriv_boltz_2(hL_j, sigma, A_j))
    Q_i2 = c * (semi_anti_deriv_boltz_2(hL_i, sigma, A_i) -
                semi_anti_deriv_boltz_2(hL_i, sigma, A_i))
    return (rsqr, a_ij, a_ji, b,
            q00, q10, q01, q11, q20, q02,
            Q_j0, Q_j1, Q_j2, Q_i0, Q_i1, Q_i2)
##########################################
