#!/usr/bin/env python
import numpy as np
from .ME_helpers import convert_sol_to_geom, sol_print_out
from .ME_zrl_evolvers import (evolver_zrl, evolver_zrl_stat, evolver_zrl_ang,
                              evolver_zrl_orient, prep_zrl_stat_evolver)
from .ME_zrl_helpers import (boltz_fact_zrl, weighted_boltz_fact_zrl,
                             fast_zrl_src_full_kl)
from .rod_motion_solver import get_rod_drag_coeff

"""@package docstring
File: choose_ME_evolver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def choose_ME_evolver(sol, vo, fs, ko, c, ks, beta, L1,
                      L2, d, visc, ODE_type='zrl'):
    """!Create a closure for ode solver

    @param sol: Array of time-dependent variables in the ODE
    @param t: time
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param d: Diameter of rods
    @param visc: Viscocity of surrounding fluid
    @param ODE_type: Which ODE to use (zrl, zrl_stat)
    @return: evolver function for ODE of interest

    """

    if ODE_type == 'zrl':
        # Get drag coefficients
        gpara1, gperp1, grot1 = get_rod_drag_coeff(visc, L1, d)
        gpara2, gperp2, grot2 = get_rod_drag_coeff(visc, L2, d)

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
                               vo, fs, ko, c, ks, beta, L1, L2, fast='fast')  # Other parameters
        return evolver_zrl_closure

    elif ODE_type == 'zrl_stat':
        # Compute geometric terms that will not change
        rsqr, a1, a2, b, q, q10, q01, q11, q20, q02 = prep_zrl_stat_evolver(
            sol, ks, beta, L1, L2)

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
                                    rsqr, a1, a2, b, q, q10, q01, q11, q20, q02,  # Pre-computed values
                                    vo, fs, ko, c, ks, beta, L1, L2)  # Other parameters
        return evolver_zrl_stat_closure

    elif ODE_type == 'zrl_ang':
        r1, r2, u1, u2 = convert_sol_to_geom(sol)
        r12 = r1 - r2

        def evolver_zrl_ang_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            r1, r2, u1, u2 = convert_sol_to_geom(sol)
            return evolver_zrl_ang(u1, u2,  # Vectors
                                   sol[12], sol[13], sol[14],  # Moments
                                   sol[15], sol[16], sol[17],
                                   gpara1, gperp1, grot1,  # Friction coefficients
                                   gpara2, gperp2, grot2,
                                   r12, vo, fs, ko, c, ks, beta, L1, L2, fast='fast')  # Other parameters
        return evolver_zrl_ang_closure

    elif ODE_type == 'zrl_orient':
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
                                      vo, fs, ko, c, ks, beta, L1, L2, fast='fast')  # Other parameters
        return evolver_zrl_orient_closure
    else:
        raise IOError('{} not a defined ODE equation for foxlink.')
