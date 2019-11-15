#!/usr/bin/env python

"""@package docstring
File: ME_gen_me_evolvers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from scipy.integrate import dblquad
from .ME_helpers import dr_dt, convert_sol_to_geom
from .ME_gen_helpers import (weighted_boltz_fact_gen,
                             boltz_fact_gen,
                             avg_force_gen_2ord)
from .ME_gen_ODEs import (du_dt_gen_2ord, dmu00_dt_gen, dmu10_dt_gen_2ord,
                          dmu11_dt_gen_2ord, dmu20_dt_gen_2ord)


def prep_me_evolver_gen_2ord(sol, co, ks, ho, beta, L_i, L_j):
    """!Calculate necessary variables to evolve the solution.

    @param sol: TODO
    @param co: TODO
    @param ks: TODO
    @return: TODO

    """

    # Convert solution entries into readable geometric variables to use in
    # derivatives
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1.0 * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    # Convert solution into readable moments to use in derivatives
    (mu00, mu10, mu01, mu11, mu20, mu02) = sol[12:18].tolist()
    # Calculate source terms (qkl) to use in derivatives
    q00 = co * dblquad(boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]  # only want val, not error
    q10 = co * dblquad(weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[1, 0, rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]
    q01 = co * dblquad(weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[0, 1, rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]
    q11 = co * dblquad(weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[1, 1, rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]
    q20 = co * dblquad(weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[2, 0, rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]
    q02 = co * dblquad(weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                       lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                       args=[0, 2, rsqr, a_ij, a_ji, b, ks, ho, beta], epsrel=1e-5)[0]
    return (r_ij, u_i, u_j,  # Vector quantities
            rsqr, a_ij, a_ji, b,  # Scalar geometric quantities
            mu00, mu10, mu01, mu11, mu20, mu02,
            q00, q10, q01, q11, q20, q02)


def me_evolver_gen_2ord(sol,
                        gpara_i, gperp_i, grot_i, gpara_j, gperp_j, grot_j,
                        vo, fs, ko, co, ks, ho, beta, L_i, L_j):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (gen) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Current solution of ODE
    @param gpara_ij: Parallel drag coefficient of rod_i
    @param gperp_i: Perpendicular drag coefficient of rod_i
    @param grot_i: Rotational drag coefficient of rod_i
    @param gpara_ji: Parallel drag coefficient of rod_i
    @param gperp_j: Perpendicular drag coefficient of rod_i
    @param grot_j: Rotational drag coefficient of rod_i
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param co: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod_i
    @param L_j: Length of rod_j
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Get variables needed to solve ODE
    (r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
     mu00, mu10, mu01, mu11, mu20, mu02,
     q00, q10, q01, q11, q20, q02) = prep_me_evolver_gen_2ord(sol, co, ks, ho,
                                                              beta, L_i, L_j)

    # Get average force of crosslinkers on rod_j
    f_ij = avg_force_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
                              mu00, mu10, mu01, mu11, mu20, mu02,
                              ks, ho)
    # Evolution of rod positions
    dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
    dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
    # Evolution of orientation vectors
    du_i = du_dt_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
                          mu10, mu11, mu20,
                          ks, ho, grot_i)
    du_j = du_dt_gen_2ord(-1. * r_ij, u_j, u_i, rsqr, a_ji, a_ij, b,  # ij-ji
                          mu01, mu11, mu02,  # kl->lk
                          ks, ho, grot_j)
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_gen(mu00, ko, q00)
    # Evoultion of first moments
    dmu10 = dmu10_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu00, mu10, mu01, mu11, mu20, mu02,
                              ko, vo, fs, ks, ho, q=q10)
    dmu01 = dmu10_dt_gen_2ord(rsqr, a_ji, a_ij, b,  # ij->ji
                              mu00, mu01, mu10, mu11, mu02, mu20,  # kl->lk
                              ko, vo, fs, ks, ho, q=q01)
    # Evolution of second moments
    dmu11 = dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu10, mu01, mu11, mu20, mu02,
                              ko, vo, fs, ks, ho, q=q11)
    dmu20 = dmu20_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu10, mu11, mu20,
                              ko, vo, fs, ks, ho, q=q20)
    dmu02 = dmu20_dt_gen_2ord(rsqr, a_ji, a_ij, b,  # ij->ji
                              mu01, mu11, mu02,  # kl->lk
                              ko, vo, fs, ks, ho, q=q02)

    dsol = np.concatenate(
        (dr_i, dr_j, du_i, du_j, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol


def me_evolver_gen_orient_2ord(sol,
                               gpara_i, gperp_i, grot_i, gpara_j, gperp_j, grot_j,
                               vo, fs, ko, co, ks, ho, beta, L_i, L_j):
    """!Calculate all time derivatives necessary to solve the moment expansion
    evolution of the Fokker-Planck equation of zero rest length (gen) crosslinkers
    bound to moving rods. d<var> is the time derivative of corresponding variable

    @param sol: Current solution of ODE
    @param gpara_ij: Parallel drag coefficient of rod_i
    @param gperp_i: Perpendicular drag coefficient of rod_i
    @param grot_i: Rotational drag coefficient of rod_i
    @param gpara_ji: Parallel drag coefficient of rod_i
    @param gperp_j: Perpendicular drag coefficient of rod_i
    @param grot_j: Rotational drag coefficient of rod_i
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param co: Effective concentration of motors in solution
    @param ks: Motor spring constant
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L_i: Length of rod_i
    @param L_j: Length of rod_j
    @return: Time-derivatives of all time varying quantities in a flattened
             array
    """
    # Get variables needed to solve ODE
    (r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
     mu00, mu10, mu01, mu11, mu20, mu02,
     q00, q10, q01, q11, q20, q02) = prep_me_evolver_gen_2ord(sol, co, ks, ho,
                                                              beta, L_i, L_j)

    # Get average force of crosslinkers on rod_j
    f_ij = avg_force_gen_2ord(r_ij, u_i, u_j, rsqr, a_ij, a_ji, b,
                              mu00, mu10, mu01, mu11, mu20, mu02,
                              ks, ho)
    # Evolution of rod positions
    dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
    dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
    # Orientations are not updated
    du_arr = np.zeros(6)
    # Evolution of zeroth moment
    dmu00 = dmu00_dt_gen(mu00, ko, q00)
    # Evoultion of first moments
    print("q10:", q10)
    print("q01:", q01)
    dmu10 = dmu10_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu00, mu10, mu01, mu11, mu20, mu02,
                              ko, vo, fs, ks, ho, q=q10)
    dmu01 = dmu10_dt_gen_2ord(rsqr, a_ji, a_ij, b,  # ij->ji
                              mu00, mu01, mu10, mu11, mu02, mu20,  # kl->lk
                              ko, vo, fs, ks, ho, q=q01)

    # Evolution of second moments
    dmu11 = dmu11_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu10, mu01, mu11, mu20, mu02,
                              ko, vo, fs, ks, ho, q=q11)
    dmu20 = dmu20_dt_gen_2ord(rsqr, a_ij, a_ji, b,
                              mu10, mu11, mu20,
                              ko, vo, fs, ks, ho, q=q20)
    dmu02 = dmu20_dt_gen_2ord(rsqr, a_ji, a_ij, b,  # ij->ji
                              mu01, mu11, mu02,  # kl->lk
                              ko, vo, fs, ks, ho, q=q02)

    dsol = np.concatenate(
        (dr_i, dr_j, du_arr, [dmu00, dmu10, dmu01, dmu11, dmu20, dmu02]))
    # Check to make sure all values are finite
    if not np.all(np.isfinite(dsol)):
        raise RuntimeError(
            'Infinity or NaN thrown in ODE solver derivatives. Current derivatives', dsol)

    return dsol
# TODO: Need to make these functions eventually <11-11-19, ARL> #
# me_evolver_gen_stat_2ord
# me_evolver_gen_ang_2ord
# me_evolver_gen_orient_2ord
# @param r_i: Center of mass postion of rod_i
# @param r_j: Center of mass position of rod_j
# @param u_i: Orientation unit vector of rod_i
# @param u_j: Orientation unit vector of rod_j
# @param mu00: Zeroth motor moment
# @param mu10: First motor moment of s_i
# @param mu01: First motor moment of s_j
# @param mu11: Second motor moment of s_i and s_j
# @param mu20: Second motor moment of s_i
# @param mu02: Second motor moment of s_j
