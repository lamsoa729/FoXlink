#!/usr/bin/env python

"""@package docstring
File: choose_me_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Function that creates closures of ODE system evolvers
"""

import numpy as np
from .me_helpers import convert_sol_to_geom, sol_print_out
from .me_zrl_evolvers import (evolver_zrl, evolver_zrl_stat, prep_zrl_evolver,
                              get_zrl_moments)
from .me_gen_evolvers import me_evolver_gen_2ord, me_evolver_gen_orient_2ord
from .rod_motion_solver import get_rod_drag_coeff


def choose_me_evolver(sol_init, slvr):
    """!Create a closure for ode solver

    @param sol: Array of time-dependent variables in the ODE
    @param t: time
    @param slvr: MomentExpansionSolver solver class
    @return: evolver function for ODE of interest

    """

    if slvr.ODE_type == 'zrl':
        # Get drag coefficients
        gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
            slvr.visc, slvr.L_i, slvr.rod_diam)
        gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
            slvr.visc, slvr.L_j, slvr.rod_diam)

        def evolver_zrl_closure(t, sol):
            """!Define the function of an ODE solver with zero length
            crosslinking proteins and moving rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl

            """
            # TODO Add verbose option
            # sol_print_out(sol)
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)

            return evolver_zrl(sol,
                               gpara_i, gperp_i, grot_i,  # Friction coefficients
                               gpara_j, gperp_j, grot_j,
                               slvr.vo, slvr.fs, slvr.ko, slvr.co,
                               slvr.ks, slvr.beta, slvr.L_i, slvr.L_j)
        return evolver_zrl_closure

    elif slvr.ODE_type == 'zrl_stat':
        # Compute geometric terms that will not change
        (rsqr, a_ij, a_ji, b,
         q00, q10, q01, q11, q20, q02) = prep_zrl_evolver(
            sol_init, slvr.co, slvr.ks, slvr.beta, slvr.L_i, slvr.L_j)

        def evolver_zrl_stat_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            # TODO Add verbose option
            # sol_print_out(sol)
            mu00, mu10, mu01, mu11, mu20, mu02 = get_zrl_moments(sol)
            # sol_print_out(sol)
            return evolver_zrl_stat(mu00, mu10, mu01, mu11, mu20, mu02,  # Moments
                                    a_ij, a_ji, b,
                                    q00, q10, q01, q11, q20, q02,
                                    slvr.vo, slvr.fs, slvr.ko, slvr.ks)  # Other parameters
        return evolver_zrl_stat_closure

    # elif slvr.ODE_type == 'zrl_ang':
    #     gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
    #         slvr.visc, slvr.L_i, slvr.rod_diam)
    #     gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
    #         slvr.visc, slvr.L_j, slvr.rod_diam)
    #     # r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)

    #     def evolver_zrl_ang_closure(t, sol):
    #         """!Define the function of an ODE solver with zero rest length
    #         crosslinking protiens and stationary rods.

    #         @param t: Time array
    #         @param sol: Solution array
    #         @return: Function to ODE zrl stat

    #         """
    #         r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    #         r_ij = r_j - r_i
    #         return evolver_zrl_ang(u_i, u_j,  # Vectors
    #                                sol[12], sol[13], sol[14],  # Moments
    #                                sol[15], sol[16], sol[17],
    #                                gpara_i, gperp_i, grot_i,  # Friction coefficients
    #                                gpara_j, gperp_j, grot_j,
    #                                r_ij, slvr.vo, slvr.fs, slvr.ko, slvr.co,
    #                                slvr.ks, slvr.beta, slvr.L_i, slvr.L_j, fast='fast')
    #     return evolver_zrl_ang_closure

    # elif slvr.ODE_type == 'zrl_orient':
    #     gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
    #         slvr.visc, slvr.L_i, slvr.rod_diam)
    #     gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
    #         slvr.visc, slvr.L_j, slvr.rod_diam)
    #     r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    #     r_ij = r_i - r_j

    #     def evolver_zrl_orient_closure(t, sol):
    #         """!Define the function of an ODE solver with zero rest length
    #         crosslinking protiens and stationary rods.

    #         @param t: Time array
    #         @param sol: Solution array
    #         @return: Function to ODE zrl stat

    #         """
    #         r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    #         return evolver_zrl_orient(r_i, r_j, u_i, u_j,  # Vectors
    #                                   # Moments
    #                                   sol[12], sol[13], sol[14],
    #                                   sol[15], sol[16], sol[17],
    #                                   gpara_i, gperp_i, grot_i,  # Friction coefficients
    #                                   gpara_j, gperp_j, grot_j,
    #                                   slvr.vo, slvr.fs, slvr.ko, slvr.co,
    #                                   slvr.ks, slvr.beta, slvr.L_i, slvr.L_j,
    #                                   fast='fast')
    #     return evolver_zrl_orient_closure

    elif slvr.ODE_type == 'gen_2ord':
        gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
            slvr.visc, slvr.L_i, slvr.rod_diam)
        gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
            slvr.visc, slvr.L_j, slvr.rod_diam)

        def me_evolver_gen_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            sol[6:9] /= np.linalg.norm(sol[6:9])
            sol[9:12] /= np.linalg.norm(sol[9:12])
            return me_evolver_gen_2ord(sol,
                                       gpara_i, gperp_i, grot_i,
                                       gpara_j, gperp_j, grot_j,
                                       slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                       slvr.ks, slvr.ho, slvr.beta,
                                       slvr.L_i, slvr.L_j)

        return me_evolver_gen_2ord_closure

    elif slvr.ODE_type == 'gen_orient_2ord':
        gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
            slvr.visc, slvr.L_i, slvr.rod_diam)
        gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
            slvr.visc, slvr.L_j, slvr.rod_diam)

        def me_evolver_gen_orient_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            return me_evolver_gen_orient_2ord(sol, gpara_i, gperp_i, grot_i,
                                              gpara_j, gperp_j, grot_j,
                                              slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                              slvr.ks, slvr.ho, slvr.beta,
                                              slvr.L_i, slvr.L_j)
        return me_evolver_gen_orient_2ord_closure

    else:
        raise IOError('{} not a defined ODE equation for foxlink.')