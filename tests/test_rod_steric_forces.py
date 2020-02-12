#!/usr/bin/env python

"""@package docstring
File: test_rod_steric_forces.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from foxlink.rod_steric_forces import find_sphero_min_dist, wca_force
import pytest


@pytest.mark.parametrize("r_i, r_j, u_i, u_j, L_i, L_j, exp_results", [
    ([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 10., 10., [[0, 1, 0], 0, 0]),
    ([0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], 10., 10., [[0, 0, 0], 0, 0]),
    ([0, 0, 0], [11, 0, 0], [1, 0, 0], [1, 0, 0], 10., 10., [[1, 0, 0], 5, -5]),
    ([0, 0, 0], [6, 1, 0], [1, 0, 0], [0, 1, 0], 10., 10., [[1, 0, 0], 5, -1]),
    ([0, 0, 0], [5, 1, 0], [1, 0, 0], [1, 0, 0], 10., 10., [[0, 1, 0], 2.5, -2.5]),
    ([0, 0, 0], [0, 4, 0], [1, 0, 0], [4. / 5., 3. / 5., 0],
     10., 10., [[0, 1, 0], -4., -5.]),
])
def test_find_sphero_min_dist(r_i, r_j, u_i, u_j, L_i, L_j, exp_results):

    r_i = np.asarray(r_i)
    r_j = np.asarray(r_j)
    u_i = np.asarray(u_i)
    u_j = np.asarray(u_j)
    min_vec_ij, l_i, l_j = find_sphero_min_dist(r_i, r_j, u_i, u_j, L_i, L_j)
    assert np.array_equal(min_vec_ij, np.asarray(exp_results[0]))
    assert l_i == exp_results[1]
    assert l_j == exp_results[2]


@pytest.mark.parametrize("dr, sigma, eps, exp_val", [
    ([2., 0, 0], 1., 1., [0, 0, 0]),
    ([1., 0, 0], 1., 1., [24., 0, 0]),
    ([1., 0, 0], 1., 2., [48., 0, 0]),
    ([.1, 0, 0], 1., 1., [479999760000000., 0, 0]),
])
def test_wca_force(dr, sigma, eps, exp_val):
    dr = np.asarray(dr)
    f_vec = wca_force(dr, sigma, eps)
    assert np.array_equal(f_vec, np.asarray(exp_val))
