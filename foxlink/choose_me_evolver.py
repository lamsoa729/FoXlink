#!/usr/bin/env python
"""
ODE Evolver Factory Module

This module provides a factory function for creating appropriate ODE evolvers
based on the solver type. It creates closures that encapsulate the specific
evolver functions with their required parameters.

Author: Adam Lamson
Email: adam.lamson@colorado.edu
"""

import numpy as np
from .me_helpers import sol_print_out
from .me_zrl_evolvers import (
    evolver_zrl,
    evolver_zrl_stat,
    evolver_zrl_bvg,
    prep_zrl_evolver,
    get_zrl_moments,
)
from .me_zrl_bound_evolvers import evolver_zrl_bound
from .me_gen_evolvers import me_evolver_gen_2ord, me_evolver_gen_orient_2ord
from .rod_motion_solver import calc_rod_drag_coeff
from .me_n_fil_evolvers import (
    me_evolver_nfil_crosslink,
    me_evolver_nfil_finite_crosslink,
)


def choose_me_evolver(sol_init, slvr):
    """
    Factory function to create appropriate ODE evolver based on solver type.

    Creates a closure that encapsulates the specific evolver function with
    its required parameters (friction coefficients, solver parameters, etc.).

    Args:
        sol_init (np.ndarray): Initial solution array for the ODE system
        slvr (MomentExpansionSolver): Solver instance containing parameters

    Returns:
        callable: Evolver function with signature evolver(t, sol) -> dsol_dt

    Raises:
        IOError: If ODE_type is not recognized
        RuntimeError: If solution contains non-finite values during evolution
    """
    ode_type = slvr.ODE_type

    # Multi-filament crosslink evolvers
    if ode_type in ["n_fil_zrl_xl", "n_fil_finite_zrl_xl"]:
        return _create_nfil_evolver(ode_type, slvr)

    # Two-filament zero rest length evolvers
    elif ode_type in ["zrl", "zrl_bvg", "zrl_bound"]:
        return _create_zrl_evolver(ode_type, slvr)

    # Stationary rod evolver
    elif ode_type == "zrl_stat":
        return _create_zrl_stat_evolver(sol_init, slvr)

    # General second-order evolvers
    elif ode_type in ["gen_2ord", "gen_orient_2ord"]:
        return _create_gen_evolver(ode_type, slvr)

    else:
        raise IOError(f"{ode_type} is not a defined ODE equation for foxlink.")


def _create_nfil_evolver(ode_type, slvr):
    """
    Create evolver for multi-filament crosslinking systems.

    Args:
        ode_type (str): Type of ODE system ("n_fil_zrl_xl" or "n_fil_finite_zrl_xl")
        slvr: Solver instance

    Returns:
        callable: Configured evolver function
    """
    # Calculate friction coefficients for each rod
    fric_coeff_arr = [
        calc_rod_drag_coeff(slvr.visc, length, slvr.rod_diam)
        for length in slvr.rod_arr[:, 6]  # rod lengths are in column 6
    ]

    if ode_type == "n_fil_zrl_xl":
        evolver_func = me_evolver_nfil_crosslink
        name = "n-filament crosslink"
    else:  # "n_fil_finite_zrl_xl"
        evolver_func = me_evolver_nfil_finite_crosslink
        name = "n-filament finite crosslink"

    def nfil_evolver_closure(t, sol):
        """
        Closure for multi-filament crosslinking evolution.

        Args:
            t (float): Current time
            sol (np.ndarray): Current solution state

        Returns:
            np.ndarray: Time derivatives of solution

        Raises:
            RuntimeError: If solution contains non-finite values
        """
        _validate_solution(sol, t, name)
        return evolver_func(sol, fric_coeff_arr, slvr.__dict__)

    return nfil_evolver_closure


def _create_zrl_evolver(ode_type, slvr):
    """
    Create evolver for two-filament zero rest length systems.

    Args:
        ode_type (str): Type of ZRL evolver
        slvr: Solver instance

    Returns:
        callable: Configured evolver function
    """
    # Calculate combined friction coefficient for two rods
    fric_coeff = calc_rod_drag_coeff(
        slvr.visc, slvr.L_i, slvr.rod_diam
    ) + calc_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam)

    # Map ODE type to evolver function and name
    evolver_map = {
        "zrl": (evolver_zrl, "zero rest length"),
        "zrl_bvg": (evolver_zrl_bvg, "zero rest length BvG"),
        "zrl_bound": (evolver_zrl_bound, "zero rest length bounded"),
    }

    evolver_func, name = evolver_map[ode_type]

    def zrl_evolver_closure(t, sol):
        """
        Closure for zero rest length evolution.

        Args:
            t (float): Current time
            sol (np.ndarray): Current solution state

        Returns:
            np.ndarray: Time derivatives of solution
        """
        _validate_solution(sol, t, name)

        # Print solution for bounded case (debugging)
        if ode_type == "zrl_bound":
            sol_print_out(sol)

        return evolver_func(sol, fric_coeff, slvr.__dict__)

    return zrl_evolver_closure


def _create_zrl_stat_evolver(sol_init, slvr):
    """
    Create evolver for stationary rod zero rest length system.

    Args:
        sol_init (np.ndarray): Initial solution for geometric calculations
        slvr: Solver instance

    Returns:
        callable: Configured evolver function
    """
    # Pre-compute geometric terms that don't change over time
    scalar_geom, q_arr = prep_zrl_evolver(sol_init, slvr.__dict__)

    def zrl_stat_evolver_closure(t, sol):
        """
        Closure for stationary zero rest length evolution.

        Args:
            t (float): Current time
            sol (np.ndarray): Current solution state (moment variables only)

        Returns:
            np.ndarray: Time derivatives of moments
        """
        # Extract moment variables from solution
        mu_kl = get_zrl_moments(sol)
        return evolver_zrl_stat(mu_kl, scalar_geom, q_arr, slvr.__dict__)

    return zrl_stat_evolver_closure


def _create_gen_evolver(ode_type, slvr):
    """
    Create evolver for general second-order systems.

    Args:
        ode_type (str): Type of general evolver
        slvr: Solver instance

    Returns:
        callable: Configured evolver function
    """
    # Calculate combined friction coefficient
    fric_coeff = calc_rod_drag_coeff(
        slvr.visc, slvr.L_i, slvr.rod_diam
    ) + calc_rod_drag_coeff(slvr.visc, slvr.L_j, slvr.rod_diam)

    if ode_type == "gen_2ord":
        evolver_func = me_evolver_gen_2ord
        name = "general 2nd order"
        normalize_orientations = True
    else:  # "gen_orient_2ord"
        evolver_func = me_evolver_gen_orient_2ord
        name = "general orientation 2nd order"
        normalize_orientations = False

    def gen_evolver_closure(t, sol):
        """
        Closure for general second-order evolution.

        Args:
            t (float): Current time
            sol (np.ndarray): Current solution state

        Returns:
            np.ndarray: Time derivatives of solution
        """
        _validate_solution(sol, t, name)

        # Normalize orientation vectors if required
        if normalize_orientations:
            sol = sol.copy()  # Don't modify input
            sol[6:9] /= np.linalg.norm(sol[6:9])  # First rod orientation
            sol[9:12] /= np.linalg.norm(sol[9:12])  # Second rod orientation

        return evolver_func(sol, fric_coeff, slvr.__dict__)

    return gen_evolver_closure


def _validate_solution(sol, t, evolver_name, verbose=False):
    """
    Validate that solution contains only finite values.

    Args:
        sol (np.ndarray): Solution array to validate
        t (float): Current time (for error reporting)
        evolver_name (str): Name of evolver (for error reporting)

    Raises:
        RuntimeError: If solution contains non-finite values
    """
    if not np.all(np.isfinite(sol)):
        raise RuntimeError(
            f"Infinity or NaN found in {evolver_name} evolver at t={t}. "
            f"Current solution: {sol}"
        )

    # Optional: Print solution for debugging
    if verbose:
        print(f"sol({t}): {sol}")
    else:
        print(f"Time {t}")
