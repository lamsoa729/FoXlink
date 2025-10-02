#!/usr/bin/env python
"""
Zero Rest Length (ZRL) Crosslinker Helper Functions

This module provides mathematical functions for calculating moments, Boltzmann factors,
and force/torque distributions for zero rest length crosslinking motors between rods.

Author: Adam Lamson
Email: adam.r.lamson@gmail.com
"""

import numpy as np
import math
import time
from numba import njit
from scipy.integrate import quad
from .me_helpers import convert_sol_to_geom
from .bivariate_gauss_helpers import fast_gauss_moment_kl
from .profiler import profile_function
from functools import lru_cache

SQRT_PI = np.sqrt(np.pi)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


@profile_function
def pair_index(i, j, n_fils):
    """Calculate linear index for rod pair (i,j)."""
    return int((2 * n_fils - i - 1) * i / 2 + j - i - 1)


@profile_function
def get_zrl_moments(sol):
    """Extract ZRL moment variables from solution vector."""
    return sol[12:18].tolist()


@profile_function
def get_unbound_and_zrl_xl_moments_for_ij(sol, i, j, n_fils):
    """Extract unbound motor count and crosslink moments for rod pair (i,j)."""
    ij = n_fils * 7 + pair_index(i, j, n_fils) * 4
    return sol[-1], sol[ij : ij + 4]


@profile_function
def get_zrl_xl_moments_for_ij(sol, i, j, n_fils):
    """Extract crosslink moments for rod pair (i,j)."""
    ij = n_fils * 7 + pair_index(i, j, n_fils) * 4
    return sol[ij : ij + 4]


@profile_function
def get_zrl_moments_and_boundary_terms(sol):
    """Extract both moments and boundary terms from solution vector."""
    return (sol[12:18].tolist(), sol[18:26].tolist())


@profile_function
def get_mu_kl_eff(mu_kl, params):
    """Calculate effective moments using bivariate Gaussian approximation."""
    if mu_kl[0] <= 0:
        return [0] * 6

    L_i = params["L_i"]
    L_j = params["L_j"]

    # Create reversed moment list for asymmetric calculations
    mu_lk = [mu_kl[0], mu_kl[2], mu_kl[1], mu_kl[3], mu_kl[5], mu_kl[4]]

    # Calculate effective moments using fast Gaussian integration
    mu00 = fast_gauss_moment_kl(L_i, L_j, mu_kl, k=0, l=0, index=0)
    mu10 = fast_gauss_moment_kl(L_j, L_i, mu_lk, k=0, l=1, index=2)
    mu01 = fast_gauss_moment_kl(L_i, L_j, mu_kl, k=0, l=1, index=2)
    mu11 = fast_gauss_moment_kl(L_i, L_j, mu_kl, k=1, l=1, index=3)
    mu20 = fast_gauss_moment_kl(L_j, L_i, mu_lk, k=0, l=2, index=5)
    mu02 = fast_gauss_moment_kl(L_i, L_j, mu_kl, k=0, l=2, index=5)

    return [mu00, mu10, mu01, mu11, mu20, mu02]


# ==============================================================================
# BOLTZMANN FACTOR CALCULATIONS (Already optimized with @njit)
# ==============================================================================


@njit
def boltz_fact_zrl(s_i, s_j, rsqr, a1, a2, b, ks, beta):
    """Boltzmann factor for zero rest length crosslinker."""
    return np.exp(
        -0.5
        * beta
        * ks
        * (rsqr + s_i**2 + s_j**2 - (2.0 * s_i * s_j * b) + 2.0 * (s_j * a2 - s_i * a1))
    )


@njit
def weighted_boltz_fact_zrl(s_i, s_j, pow1, pow2, rsqr, a1, a2, b, ks, beta):
    """Weighted Boltzmann factor (s_i^pow1 * s_j^pow2 * boltz_fact)."""
    return (
        np.power(s_i, pow1)
        * np.power(s_j, pow2)
        * np.exp(
            -0.5
            * beta
            * ks
            * (
                rsqr
                + s_i**2
                + s_j**2
                - (2.0 * s_i * s_j * b)
                + 2.0 * (s_j * a2 - s_i * a1)
            )
        )
    )


# ==============================================================================
# SEMI-ANALYTICAL INTEGRATION FUNCTIONS (Already optimized with @njit)
# ==============================================================================


@njit
def semi_anti_deriv_boltz_0(L, sigma, A):
    """Semi-analytical integration for zeroth moment."""
    return (0.5 * SQRT_PI * sigma) * math.erf((L + A) / sigma)


@njit
def semi_anti_deriv_boltz_1(L, sigma, A):
    """Semi-analytical integration for first moment."""
    B = (L + A) / sigma
    return (-0.5 * sigma) * (sigma * np.exp(-1.0 * B * B) + (A * SQRT_PI * math.erf(B)))


@njit
def semi_anti_deriv_boltz_2(L, sigma, A):
    """Semi-analytical integration for second moment."""
    B = (L + A) / sigma
    return (0.25 * sigma) * (
        2.0 * sigma * (A - L) * np.exp(-1.0 * B * B)
        + (((2.0 * A * A) + (sigma * sigma)) * SQRT_PI) * math.erf(B)
    )


@njit
def semi_anti_deriv_boltz_3(L, sigma, A):
    """Semi-analytical integration for third moment."""
    B = (L + A) / sigma
    return (-0.25 * sigma) * (
        (2.0 * sigma * (A * A - A * L + L * L + sigma * sigma) * np.exp(-1.0 * B * B))
        + ((2.0 * A * A) + 3.0 * (sigma * sigma)) * A * SQRT_PI * math.erf(B)
    )


# ==============================================================================
# INTEGRAND FUNCTIONS (Already optimized with @njit)
# ==============================================================================


@njit
def fast_zrl_src_integrand_l0(s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """Fast calculation of source integrand for l=0."""
    A = -1.0 * (a_ji + (b * s_i))
    exponent = -1.0 * (rsqr + s_i * (s_i - 2.0 * a_ij) - (A * A)) / (sigma * sigma)

    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_0(-0.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_0(0.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_l1(s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """Fast calculation of source integrand for l=1."""
    A = -1.0 * (a_ji + (b * s_i))
    exponent = -1.0 * (rsqr + s_i * (s_i - 2.0 * a_ij) - (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_1(-0.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_1(0.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


@njit
def fast_zrl_src_integrand_l2(s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """Fast calculation of source integrand for l=2."""
    A = -1.0 * (a_ji + (b * s_i))
    exponent = -1.0 * (rsqr + s_i * (s_i - 2.0 * a_ij) - (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_2(-0.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_2(0.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


def fast_zrl_src_integrand_l3(s_i, L_j, rsqr, a_ij, a_ji, b, sigma, k=0):
    """Fast calculation of source integrand for l=3."""
    A = -1.0 * (a_ji + (b * s_i))
    exponent = -1.0 * (rsqr + s_i * (s_i - 2.0 * a_ij) - (A * A)) / (sigma * sigma)
    pre_fact = np.power(s_i, k) * np.exp(exponent)
    I_m = semi_anti_deriv_boltz_3(-0.5 * L_j, sigma, A)
    I_p = semi_anti_deriv_boltz_3(0.5 * L_j, sigma, A)
    return pre_fact * (I_p - I_m)


# ==============================================================================
# SOURCE TERM CALCULATIONS (This is likely your bottleneck!)
# ==============================================================================
# Simple cache using Python's built-in LRU cache
# @lru_cache(maxsize=4096)  # Adjust maxsize as needed
def fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k, l):
    """
    Cached version of integral calculation with LRU eviction.

    Note: All parameters must be hashable for LRU cache to work.
    """
    integrand_map = {
        0: fast_zrl_src_integrand_l0,
        1: fast_zrl_src_integrand_l1,
        2: fast_zrl_src_integrand_l2,
    }

    if l not in integrand_map:
        raise RuntimeError(f"{l}-order derivatives not implemented.")

    integrand = integrand_map[l]
    sigma = np.sqrt(2.0 / (ks * beta))

    result, error = quad(
        integrand, -0.5 * L_i, 0.5 * L_i, args=(L_j, rsqr, a_ij, a_ji, b, sigma, k)
    )
    return result


@profile_function
def _fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=0):
    """
    Public interface that rounds parameters for cache key stability.
    """
    # Round parameters to avoid floating-point precision issues in cache keys
    cache_key = (
        round(float(L_i), 6),
        round(float(L_j), 6),
        round(float(rsqr), 6),
        round(float(a_ij), 6),
        round(float(a_ji), 6),
        round(float(b), 6),
        round(float(ks), 8),
        round(float(beta), 8),
        int(k),
        int(l),
    )

    return _cached_zrl_src_kl(*cache_key)


# ==============================================================================
# EVOLVER PREPARATION FUNCTIONS
# ==============================================================================


@njit
def get_Qj_params(s_i, L_j, a_ji, b, ks, beta):
    """Calculate parameters for Q_j boundary term calculations."""
    hL_j = 0.5 * L_j
    sigma = np.sqrt(2.0 / (ks * beta))
    A_j = -1.0 * (a_ji + (b * s_i))
    return hL_j, sigma, A_j


@profile_function
def prep_zrl_nfil_evolver(r_i, u_i, L_i, r_j, u_j, L_j, params):
    """Prepare geometric and source terms for N-filament ZRL evolver."""
    ks = params["ks"]
    beta = params["beta"]
    c = params["co"] / params["volume"]

    # Length scale of crosslinker
    lx = np.sqrt(0.5 * ks * beta)

    # Geometric calculations (fast)
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1.0 * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    dist_lim = 0.5 * (L_i + L_j) + (5 * lx)
    # Check if rods are out of range
    if rsqr > (dist_lim**2):
        q00, q10, q01, q11 = (0.0, 0.0, 0.0, 0.0)

    else:
        # Source term calculations (potentially slow due to quad integration)
        q00 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=0)
        q10 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji, a_ij, b, ks, beta, k=0, l=1)
        q01 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=1)
        q11 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=1, l=1)

    return (rsqr, a_ij, a_ji, b), (q00, q10, q01, q11)


@profile_function
def prep_zrl_evolver(sol, params):
    """Prepare all terms needed for standard ZRL evolver."""
    r_i, r_j, u_i, u_j = convert_sol_to_geom(sol)
    c = params["co"]
    L_i, L_j = params["L_i"], params["L_j"]
    ks = params["ks"]
    beta = params["beta"]

    # Geometric calculations
    r_ij = r_j - r_i
    rsqr = np.dot(r_ij, r_ij)
    a_ij = np.dot(r_ij, u_i)
    a_ji = -1.0 * np.dot(r_ij, u_j)
    b = np.dot(u_i, u_j)

    # Multiple quad integrations - this is expensive!
    q00 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=0)
    q10 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji, a_ij, b, ks, beta, k=0, l=1)
    q01 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=1)
    q11 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=1, l=1)
    q20 = c * fast_zrl_src_kl(L_j, L_i, rsqr, a_ji, a_ij, b, ks, beta, k=0, l=2)
    q02 = c * fast_zrl_src_kl(L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=0, l=2)

    return (rsqr, a_ij, a_ji, b), (q00, q10, q01, q11, q20, q02)


@profile_function
def prep_zrl_bound_evolver(sol, params):
    """Prepare terms for bounded ZRL evolver including boundary contributions."""
    c = params["co"]
    L_i, L_j = params["L_i"], params["L_j"]
    ks = params["ks"]
    beta = params["beta"]

    # Get standard terms (expensive due to multiple quad calls)
    (scalar_geom, q_arr) = prep_zrl_evolver(sol, params)
    (rsqr, a_ij, a_ji, b) = scalar_geom

    # Calculate boundary terms (more function calls)
    hL_j, sigma, A_j = get_Qj_params(0.5 * L_i, L_j, a_ji, b, ks, beta)
    hL_i, sigma, A_i = get_Qj_params(hL_j, L_i, a_ij, b, ks, beta)

    Q0_j = c * fast_zrl_src_integrand_l0(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q0_i = c * fast_zrl_src_integrand_l0(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q1_j = c * fast_zrl_src_integrand_l1(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q1_i = c * fast_zrl_src_integrand_l1(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q2_j = c * fast_zrl_src_integrand_l2(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q2_i = c * fast_zrl_src_integrand_l2(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)
    Q3_j = c * fast_zrl_src_integrand_l3(hL_i, L_j, rsqr, a_ij, a_ji, b, sigma)
    Q3_i = c * fast_zrl_src_integrand_l3(hL_j, L_i, rsqr, a_ji, a_ij, b, sigma)

    return (scalar_geom, q_arr, (Q0_j, Q0_i, Q1_j, Q1_i, Q2_j, Q2_i, Q3_j, Q3_i))


# ==============================================================================
# FORCE AND TORQUE CALCULATIONS (Already optimized with @njit)
# ==============================================================================


@njit
def avg_force_zrl(r_ij, u_i, u_j, mu00, mu10, mu01, ks):
    """Calculate average force on rod j from ZRL crosslinkers."""
    return -ks * (r_ij * mu00 + mu01 * u_j - mu10 * u_i)


@njit  # Added @njit for consistency
def avg_torque_zrl(r_ij, u_i, u_j, mu10, mu11, ks):
    """Calculate average torque on rod i from ZRL crosslinkers."""
    return ks * (np.cross(u_i, r_ij) * mu10 + np.cross(u_i, u_j) * mu11)
