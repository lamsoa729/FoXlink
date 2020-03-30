#!/usr/bin/env python

"""@package docstring
File: bivariate_gauss_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from math import erf
import numpy as np
from numba import njit
from scipy.integrate import quad

SQRT_PI = np.sqrt(np.pi)  # Reduce the number of sqrts you need to do


@njit
def bivariate_gauss(s_i, s_j, mu00, mu10, mu01, sig_i, sig_j, nu, gamma):
    """!Function for bivariate gaussian distribution with normalized moments,
    variances, and correlation parameters.

    @param mu00: TODO
    @param mu10: TODO
    @param mu01: TODO
    @param mu11: TODO
    @param mu20: TODO
    @param mu02: TODO
    @return: TODO

    """
    pre_fact = .5 * gamma * mu00 / (np.pi * sig_i * sig_j)
    x = (s_i - mu10) / sig_i  # Scaled and translated s_i
    y = (s_j - mu01) / sig_j  # Scaled and translated s_j
    val = np.exp(-.5 * gamma * gamma * (x * x + y * y - (2. * nu * x * y)))
    return pre_fact * val


@njit
def weighted_bivariate_gauss(
        s_i, s_j, mu00, mu10, mu01, sig_i, sig_j, nu, gamma, k=0, l=0):
    distr = bivariate_gauss(s_i, s_j, mu00, mu10, mu01,
                            sig_i, sig_j, nu, gamma)
    return np.power(s_i, k) * np.power(s_j, l) * distr


def convert_moments_to_gauss_vars(mu_kl):
    """!TODO: Docstring for convert_to_gauss_vars.

    @param arg1: TODO
    @return: TODO

    """
    if mu_kl[0] <= 0:
        return [0] * 6
    (mu10_bar, mu01_bar, mu11_bar, mu20_bar, mu02_bar) = (mu_kl[1] / mu_kl[0],
                                                          mu_kl[2] / mu_kl[0],
                                                          mu_kl[3] / mu_kl[0],
                                                          mu_kl[4] / mu_kl[0],
                                                          mu_kl[5] / mu_kl[0])
    sigma_i = np.sqrt(mu20_bar - (mu10_bar * mu10_bar))
    sigma_j = np.sqrt(mu02_bar - (mu01_bar * mu01_bar))
    print(sigma_i)
    if sigma_i <= 0 or np.isnan(sigma_i):
        sigma_i = 1e-12
    if sigma_j <= 0 or np.isnan(sigma_j):
        sigma_j = 1e-12

    nu = (mu11_bar - (mu10_bar * mu01_bar)) / (sigma_i * sigma_j)
    if nu >= 1.:
        nu = 1. - 1e-12
    elif nu <= -1.:
        nu = 1e-12 - 1.

    gamma = 1. / np.sqrt(2. * (1. - nu * nu))
    return (mu10_bar, mu01_bar, sigma_i, sigma_j, nu, gamma)


@njit
def semi_anti_deriv_gauss_0(s_i, s_j, sigma_j, mu01, nu, gamma):
    """!Fast calculation of the s_j integral of the source term for the zeroth
    moment.

    @param s_i:
    @param s_j:
    @param mu01:
    @param gamma:
    @param nu:
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    if gamma == 0.:
        return 0.
    s = (s_j - nu * s_i) * gamma
    return (.5 * SQRT_PI / gamma) * erf(s)


@njit
def semi_anti_deriv_gauss_1(s_i, s_j, sigma_j, mu01, nu, gamma):
    """!Fast calculation of the s_j integral of the source term for the first
    moment.

    @param s_i:
    @param s_j:
    @param mu01:
    @param gamma:
    @param nu:
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    if gamma == 0.:
        return 0.
    s = (s_j - nu * s_i) * gamma
    return ((-sigma_j * np.exp(-1. * s * s)
             + (SQRT_PI * gamma) * (mu01 + sigma_j * nu * s_i) * erf(s))
            / (2. * gamma * gamma))


@njit
def semi_anti_deriv_gauss_2(s_i, s_j, sigma_j, mu01, nu, gamma):
    """!Fast calculation of the s_j integral of the source term for the second
    moment.

    @param s_i:
    @param s_j:
    @param mu01:
    @param gamma:
    @param nu:
    @return: One term in the anti-derivative of the boltzman factor integrated over s_j

    """
    if gamma == 0.:
        return 0.
    g2 = gamma * gamma
    s = (s_j - nu * s_i) * gamma
    return ((-2. * gamma * sigma_j * (s_j + nu * s_i) * np.exp(-1. * s * s)
             + SQRT_PI * (2. * g2 * mu01 * mu01
                          + sigma_j * sigma_j
                          + 2 * g2 * sigma_j * nu * s_i *
                          (2 * mu01 + sigma_j * nu * s_i)
                          * (2. * mu01 + sigma_j * nu * s_i)) * erf(s))
            / (4. * g2 * gamma))


@njit
def fast_gauss_integrand_l0(
        s_i, L_j, sigma_i, sigma_j, mu10, mu01, nu, gamma, k):
    """!TODO: Docstring for fast_zrl_src_integrand_k0.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    exponent = -1. * gamma * gamma * s_i * s_i * (1. + nu * nu)
    s_i_scale = sigma_i * s_i + mu10

    pre_fact = np.power(s_i_scale, k) * np.exp(exponent)
    upper_bound = (0.5 * L_j - mu01) / sigma_j
    lower_bound = (-.5 * L_j - mu01) / sigma_j

    I_m = semi_anti_deriv_gauss_0(s_i, lower_bound, sigma_j, mu01, nu, gamma)
    I_p = semi_anti_deriv_gauss_0(s_i, upper_bound, sigma_j, mu01, nu, gamma)
    return pre_fact * (I_p - I_m)


@njit
def fast_gauss_integrand_l1(
        s_i, L_j, sigma_i, sigma_j, mu10, mu01, nu, gamma, k):
    """!TODO: Docstring for fast_zrl_src_integrand_k1.

    @param s_i: TODO
    @param L_j: TODO
    @param rsqr: TODO
    @param a_ij: TODO
    @param a_ji: TODO
    @param b: TODO
    @param sigma: TODO
    @param k: TODO
    @return: TODO

    """
    exponent = -1. * gamma * gamma * s_i * s_i * (1. + nu * nu)
    s_i_scale = sigma_i * s_i + mu10
    pre_fact = np.power(s_i_scale, k) * np.exp(exponent)
    upper_bound = (0.5 * L_j - mu01) / sigma_j
    lower_bound = (-.5 * L_j - mu01) / sigma_j

    I_m = semi_anti_deriv_gauss_1(s_i, lower_bound, sigma_j, mu01, nu, gamma)
    I_p = semi_anti_deriv_gauss_1(s_i, upper_bound, sigma_j, mu01, nu, gamma)
    return pre_fact * (I_p - I_m)


@njit
def fast_gauss_integrand_l2(
        s_i, L_j, sigma_i, sigma_j, mu10, mu01, nu, gamma, k):
    """!TODO: Docstring for fast_zrl_src_integrand_k1.

    @param s_i:
    @param L_j:
    @param sigma_i:
    @param sigma_j:
    @param mu10:
    @param mu01:
    @param nu:
    @param gamma:
    @return: TODO

    """
    exponent = -1. * gamma * gamma * s_i * s_i * (1. + nu * nu)
    s_i_scale = sigma_i * s_i + mu10
    pre_fact = np.power(s_i_scale, k) * np.exp(exponent)
    upper_bound = (0.5 * L_j - mu01) / sigma_j
    lower_bound = (-.5 * L_j - mu01) / sigma_j

    I_m = semi_anti_deriv_gauss_2(s_i, lower_bound, sigma_j, mu01, gamma, nu)
    I_p = semi_anti_deriv_gauss_2(s_i, upper_bound, sigma_j, mu01, gamma, nu)
    return pre_fact * (I_p - I_m)


def fast_gauss_moment_kl(L_i, L_j, mu_kl, k=0, l=0, index=0):
    """!TODO: Docstring for fast_zrl_src_kl

    @param L_i:
    @param L_j:
    @param mu_kl: List of moments
    @param k:
    @param l:
    @return: TODO

    """
    if l == 0:
        integrand = fast_gauss_integrand_l0
    elif l == 1:
        integrand = fast_gauss_integrand_l1
    elif l == 2:
        integrand = fast_gauss_integrand_l2
    else:
        raise RuntimeError(
            "{}-order derivatives have not been implemented for fast source solver.".format(l))
    (mu10_bar, mu01_bar,
     sigma_i, sigma_j,
     nu, gamma) = convert_moments_to_gauss_vars(mu_kl)

    upper_bound = (0.5 * L_i - mu10_bar) / sigma_i
    lower_bound = (-.5 * L_i - mu10_bar) / sigma_i
    mukl_gauss, e = quad(integrand, lower_bound, upper_bound,
                         args=(L_j, sigma_i, sigma_j, mu10_bar, mu01_bar,
                               nu, gamma, k),
                         epsrel=1e-8)
    return (np.sqrt(.5) * mu_kl[0] * gamma / np.pi) * mukl_gauss
