#!/usr/bin/env python

"""@package docstring
File: pde_steady_state.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
import yaml
from scipy.special import erf
from scipy.integrate import quad


def pde_steady_state_antipara(s_i, s_j, y, p_dict):
    L, ks, fs, ko, co, vo, beta = (p_dict['L1'], p_dict['ks'], p_dict['fs'],
                                   p_dict['ko'], p_dict['co'], p_dict['vo'],
                                   p_dict['beta'])
    xi = s_i + s_j
    lo = 2. * vo / ko
    ls = fs / ks
    a = beta * ks
    alpha = co * np.exp(-.5 * a * y * y) / lo

    # Create closure functions for different regions of f-v relation
    def apara_region_1(z):
        pre_fact = alpha * np.sqrt(.5 * np.pi / a) * \
            np.exp(.5 / (a * lo * lo) - (z / lo))
        integral = erf((a * lo * z - 1.) / (lo * np.sqrt(2. * a))) - \
            erf((-a * lo * L - 1.) / (lo * np.sqrt(2. * a)))
        return pre_fact * integral

    def apara_region_2(z):
        psi_0 = np.power(1. - (z / ls), (ls / lo) - 1.) * apara_region_1(0)
        pre_fact = alpha * ls * np.power(ls - z, (ls / lo) - 1.)

        def integrand(x):
            return np.power(ls - x, -ls / lo) * np.exp(-.5 * a * x * x)
        integral = np.frompyfunc(lambda x: quad(integrand, 0, x)[0], 1, 1)
        return pre_fact * integral(z) + psi_0

    def apara_region_3(z):
        return co * np.exp(-.5 * a * (z * z + y * y))
    sol = np.piecewise(
        xi, [xi <= 0, (xi > 0) & (xi <= ls), ls < xi],
        [apara_region_1, apara_region_2, apara_region_3])
    return sol


def pde_error_measure(h5_data):
    s_i, s_j = h5_data['rod_data/s1'][:-1], h5_data['rod_data/s2'][:-1]
    sol = h5_data['xl_data/xl_distr'][:-1, :-1, -1]
    y = h5_data['rod_data/R2_pos'][-1, 2] - h5_data['rod_data/R1_pos'][-1, 2]
    p_dict = yaml.safe_load(h5_data.attrs['params'])
    ds = p_dict['ds']
    S_i, S_j = np.meshgrid(s_i, s_j, indexing='ij')
    sol_comp = pde_steady_state_antipara(S_i, S_j, y, p_dict)
    # print(sol_comp)

    comp = np.absolute(sol - sol_comp)
    return np.sum(comp) * ds * ds
