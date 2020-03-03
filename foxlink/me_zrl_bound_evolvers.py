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
                          dmu11_dt_zrl, dmu20_dt_zrl, dBl_j_dt_zrl)
from .me_zrl_helpers import (avg_force_zrl, fast_zrl_src_integrand_l0,
                             fast_zrl_src_integrand_l1,
                             fast_zrl_src_integrand_l2,
                             fast_zrl_src_integrand_l3)
from .rod_steric_forces import calc_wca_force_torque
from .me_zrl_evolvers import prep_zrl_evolver


def get_zrl_moments_and_boundary_terms(sol):
    """!Get the moments from the solution vector of solve_ivp

    @param sol: Solution vector
    @return: Moments of the solution vector

    """
    return sol[12:26].tolist()


def get_Qj_params(s_i, L_j, a_ji, b, ks, beta):
    hL_j = .5 * L_j
    sigma = np.sqrt(2. / (ks * beta))
    A_j = -1. * (a_ji + (b * s_i))
    return hL_j, sigma, A_j


def prep_zrl_bound_evolver(sol, c, ks, beta, L_i, L_j):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param arg1: TODO
    @return: TODO

    """
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(sol, c, ks,
                                                      beta, L_i, L_j)

    hL_j, sigma, A_j = get_Qj_params(.5 * L_i, L_j, a_ji, b, ks, beta)
    hL_i, sigma, A_i = get_Qj_params(hL_j, L_i, a_ij, b, ks, beta)
    Q0_j = c * fast_zrl_src_integrand_l0(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q0_i = c * fast_zrl_src_integrand_l0(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q1_j = c * fast_zrl_src_integrand_l1(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q1_i = c * fast_zrl_src_integrand_l1(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q2_j = c * fast_zrl_src_integrand_l2(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q2_i = c * fast_zrl_src_integrand_l2(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q3_j = c * fast_zrl_src_integrand_l3(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q3_i = c * fast_zrl_src_integrand_l3(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    return (rsqr, a_ij, a_ji, b,
            q00, q10, q01, q11, q20, q02,
            Q0_j, Q0_i, Q1_j, Q1_i, Q2_j, Q2_i, Q3_j, Q3_i)


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
    # Define useful parameters for functions
    hL_i, hL_j = .5 * L_i, .5 * L_j
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    (rsqr, a_ij, a_ji, b,
     q00, q10, q01, q11, q20, q02,
     Q0_j, Q0_i, Q1_j, Q1_i, Q2_j, Q2_i, Q3_j, Q3_i) = prep_zrl_bound_evolver(
        sol, c, ks, beta, L_i, L_j)
    (mu00, mu10, mu01, mu11, mu20, mu02,
     B0_j, B0_i, B1_j, B1_i, B2_j, B2_i, B3_j, B3_i) = get_zrl_moments_and_boundary_terms(sol)
    # if mu00 < 0.:
    #     mu00 = 0.
    #     # sol[12] = 0.
    # if mu20 < 0.:
    #     mu20 = 0.
    #     # sol[16] = 0.
    # if mu02 < 0.:
    #     mu02 = 0.
    #     # sol[17] = 0.

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
    dmu00 = dmu00_dt_zrl(mu00, a_ij, a_ji, b, hL_i, hL_j, ko, vo, kappa, q00,
                         B0_j, B0_i, B1_j, B1_i)
    # Evoultion of first moments
    dmu10 = dmu10_dt_zrl(mu00, mu10, mu01, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q10, B0_j, B1_j, B1_i, B2_i)
    dmu01 = dmu10_dt_zrl(mu00, mu01, mu10, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q01, B0_i, B1_i, B1_j, B2_j)
    # Evolution of second moments
    dmu11 = dmu11_dt_zrl(mu10, mu01, mu11, mu20, mu02, a_ij, a_ji, b,
                         hL_j, hL_i, ko, vo, kappa, q11, B1_j, B1_i, B2_j, B2_i)
    dmu20 = dmu20_dt_zrl(mu10, mu11, mu20, a_ij, a_ji, b, hL_i, hL_j,
                         ko, vo, kappa, q20, B0_j, B1_j, B2_i, B3_i)
    dmu02 = dmu20_dt_zrl(mu01, mu11, mu02, a_ji, a_ij, b, hL_j, hL_i,
                         ko, vo, kappa, q02, B0_i, B1_i, B2_j, B3_j)

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
    dsol = np.concatenate(
        (dr_i, dr_j, du_i, du_j,
            [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02,
             dB0_j, dB0_i, dB1_j, dB1_i, dB2_j, dB2_i, dB3_j, dB3_i]))
    return dsol

##########################################
