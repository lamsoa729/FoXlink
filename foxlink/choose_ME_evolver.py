#!/usr/bin/env python

"""@package docstring
File: choose_ME_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from .ME_helpers import convert_sol_to_geom
from .ME_zrl_evolvers import (evolver_zrl, evolver_zrl_stat, evolver_zrl_ang,
                              evolver_zrl_orient, prep_zrl_stat_evolver)
from .ME_gen_evolvers import me_evolver_gen_2ord, me_evolver_gen_orient_2ord
from .rod_motion_solver import get_rod_drag_coeff


def choose_ME_evolver(sol, slvr):
    """!Create a closure for ode solver

    @param sol: Array of time-dependent variables in the ODE
    @param t: time
    @param slvr: MomentExpansionSolver solver class
    @return: evolver function for ODE of interest

    """

    if slvr.ODE_type == 'zrl':
        # Get drag coefficients
        gpara1, gperp1, grot1 = get_rod_drag_coeff(
            slvr.visc, slvr.L1, slvr.rod_diam)
        gpara2, gperp2, grot2 = get_rod_drag_coeff(
            slvr.visc, slvr.L2, slvr.rod_diam)

        def evolver_zrl_closure(t, sol):
            """!Define the function of an ODE solver with zero length
            crosslinking proteins and moving rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl

            """
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)

            r1, r2, u1, u2 = convert_sol_to_geom(sol)
            # sol_print_out(sol)
            return evolver_zrl(r1, r2, u1, u2,  # Vectors
                               sol[12], sol[13], sol[14],  # Moments
                               sol[15], sol[16], sol[17],
                               gpara1, gperp1, grot1,  # Friction coefficients
                               gpara2, gperp2, grot2,
                               slvr.vo, slvr.fs, slvr.ko, slvr.co,
                               slvr.ks, slvr.beta, slvr.L1, slvr.L2,
                               fast='fast')
        return evolver_zrl_closure

    elif slvr.ODE_type == 'zrl_stat':
        # Compute geometric terms that will not change
        rsqr, a1, a2, b, q00, q10, q01, q11, q20, q02 = prep_zrl_stat_evolver(
            sol, slvr.ks, slvr.beta, slvr.L1, slvr.L2)

        def evolver_zrl_stat_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            # sol_print_out(sol)
            return evolver_zrl_stat(sol[12], sol[13], sol[14],  # Moments
                                    sol[15], sol[16], sol[17],
                                    rsqr, a1, a2, b, q00, q10, q01, q11, q20, q02,
                                    slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                    slvr.ks, slvr.beta, slvr.L1, slvr.L2)  # Other parameters
        return evolver_zrl_stat_closure

    elif slvr.ODE_type == 'zrl_ang':
        gpara1, gperp1, grot1 = get_rod_drag_coeff(
            slvr.visc, slvr.L1, slvr.rod_diam)
        gpara2, gperp2, grot2 = get_rod_drag_coeff(
            slvr.visc, slvr.L2, slvr.rod_diam)
        r1, r2, u1, u2 = convert_sol_to_geom(sol)

        def evolver_zrl_ang_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            r1, r2, u1, u2 = convert_sol_to_geom(sol)
            r12 = r2 - r1
            return evolver_zrl_ang(u1, u2,  # Vectors
                                   sol[12], sol[13], sol[14],  # Moments
                                   sol[15], sol[16], sol[17],
                                   gpara1, gperp1, grot1,  # Friction coefficients
                                   gpara2, gperp2, grot2,
                                   r12, slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                   slvr.ks, slvr.beta, slvr.L1, slvr.L2, fast='fast')
        return evolver_zrl_ang_closure

    elif slvr.ODE_type == 'zrl_orient':
        gpara1, gperp1, grot1 = get_rod_drag_coeff(
            slvr.visc, slvr.L1, slvr.rod_diam)
        gpara2, gperp2, grot2 = get_rod_drag_coeff(
            slvr.visc, slvr.L2, slvr.rod_diam)
        r1, r2, u1, u2 = convert_sol_to_geom(sol)
        r12 = r1 - r2

        def evolver_zrl_orient_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            r1, r2, u1, u2 = convert_sol_to_geom(sol)
            return evolver_zrl_orient(r1, r2, u1, u2,  # Vectors
                                      # Moments
                                      sol[12], sol[13], sol[14],
                                      sol[15], sol[16], sol[17],
                                      gpara1, gperp1, grot1,  # Friction coefficients
                                      gpara2, gperp2, grot2,
                                      slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                      slvr.ks, slvr.beta, slvr.L1, slvr.L2,
                                      fast='fast')
        return evolver_zrl_orient_closure

    elif slvr.ODE_type == 'gen_2ord':
        gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
            slvr.visc, slvr.L1, slvr.rod_diam)
        gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
            slvr.visc, slvr.L2, slvr.rod_diam)

        def me_evolver_gen_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            sol[6:9] /= np.linalg.norm(sol[6:9])
            sol[9:12] /= np.linalg.norm(sol[9:12])
            return me_evolver_gen_2ord(sol, gpara_i, gperp_i, grot_i,
                                       gpara_j, gperp_j, grot_j,
                                       slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                       slvr.ks, slvr.ho, slvr.beta,
                                       slvr.L1, slvr.L2)

        return me_evolver_gen_2ord_closure

    elif slvr.ODE_type == 'gen_orient_2ord':
        gpara_i, gperp_i, grot_i = get_rod_drag_coeff(
            slvr.visc, slvr.L1, slvr.rod_diam)
        gpara_j, gperp_j, grot_j = get_rod_drag_coeff(
            slvr.visc, slvr.L2, slvr.rod_diam)

        def me_evolver_gen_orient_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            return me_evolver_gen_orient_2ord(sol, gpara_i, gperp_i, grot_i,
                                              gpara_j, gperp_j, grot_j,
                                              slvr.vo, slvr.fs, slvr.ko, slvr.co,
                                              slvr.ks, slvr.ho, slvr.beta,
                                              slvr.L1, slvr.L2)
        return me_evolver_gen_orient_2ord_closure

    else:
        raise IOError('{} not a defined ODE equation for foxlink.')
