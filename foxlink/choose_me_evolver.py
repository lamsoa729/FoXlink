#!/usr/bin/env python

"""@package docstring
File: choose_me_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Function that creates closures of ODE system evolvers
"""

import numpy as np
from .me_helpers import sol_print_out
from .me_zrl_evolvers import (evolver_zrl, evolver_zrl_stat, evolver_zrl_bvg,
                              prep_zrl_evolver, get_zrl_moments)
from .me_zrl_odes import calc_moment_derivs_zrl
from .me_zrl_bound_evolvers import evolver_zrl_bound
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
        fric_coeff = (get_rod_drag_coeff(slvr.visc, slvr.L_i, slvr.rod_diam) +
                      get_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam))

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
            return evolver_zrl(sol, fric_coeff, slvr.__dict__)
        return evolver_zrl_closure

    if slvr.ODE_type == 'zrl_bvg':
        # Get drag coefficients
        fric_coeff = (get_rod_drag_coeff(slvr.visc, slvr.L_i, slvr.rod_diam) +
                      get_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam))

        def evolver_zrl_bvg_closure(t, sol):
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
            return evolver_zrl_bvg(sol, fric_coeff, slvr.__dict__)
        return evolver_zrl_bvg_closure

    if slvr.ODE_type == 'zrl_bound':
        # Get drag coefficients
        fric_coeff = (get_rod_drag_coeff(slvr.visc, slvr.L_i, slvr.rod_diam) +
                      get_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam))

        def evolver_zrl_bound_closure(t, sol):
            """!Define the function of an ODE solver with zero length
            crosslinking proteins and moving rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl

            """
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)

            sol_print_out(sol)
            return evolver_zrl_bound(sol, fric_coeff, slvr.__dict__)
        return evolver_zrl_bound_closure

    if slvr.ODE_type == 'zrl_stat':
        # Compute geometric terms that will not change
        scalar_geom, q_arr = prep_zrl_evolver(sol_init, slvr.__dict__)

        def evolver_zrl_stat_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            # TODO Add verbose option
            # sol_print_out(sol)
            mu_kl = get_zrl_moments(sol)
            return evolver_zrl_stat(mu_kl, scalar_geom, q_arr, slvr.__dict__)
        return evolver_zrl_stat_closure

    if slvr.ODE_type == 'gen_2ord':
        fric_coeff = (get_rod_drag_coeff(slvr.visc, slvr.L_i, slvr.rod_diam) +
                      get_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam))

        def me_evolver_gen_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            sol[6:9] /= np.linalg.norm(sol[6:9])
            sol[9:12] /= np.linalg.norm(sol[9:12])
            return me_evolver_gen_2ord(sol, fric_coeff, slvr.__dict__)

        return me_evolver_gen_2ord_closure

    if slvr.ODE_type == 'gen_orient_2ord':
        fric_coeff = (get_rod_drag_coeff(slvr.visc, slvr.L_i, slvr.rod_diam) +
                      get_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam))

        def me_evolver_gen_orient_2ord_closure(t, sol):
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)
            print("sol({}):".format(t), sol)

            return me_evolver_gen_orient_2ord(sol, fric_coeff, slvr.__dict__)
        return me_evolver_gen_orient_2ord_closure

    raise IOError('{} not a defined ODE equation for foxlink.')
