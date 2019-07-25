#!/usr/bin/env python
import pytest
import numpy as np
from math import exp
from foxlink.FP_helpers import make_force_dep_velocity_mat, boltz_fact_mat


"""@package docstring
File: test_fp_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


# @pytest.fixture
# def force_velocity_params():
#     """ Standard parameters for testing force dependent velocity matrix

#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     f_mat = np.ones((6, 6, 3))
#     u_vec = [0, 1., 0]
#     fs = 1.
#     vo = 1.
#     return (f_mat, u_vec, fs, vo)

@pytest.mark.parametrize("force, u_vec, fs, vo, expected_v", [
    (4., [0, 1, 0], 1., 1., 1.),
    (4., [1, 0, 0], 1., 1., 1.),
    (4., [0, 0, 1], 1., 1., 1.),
    (-.5, [0, 1, 0], 1., 1., .50),
    (-4, [0, 1, 0], 1., 1., 0),
])
def test_force_dep_velocity_mat(force, u_vec, fs, vo, expected_v):
    """!TODO: Docstring for test_force_dep_velocity_mat.

    @param force: TODO
    @param u_vec: TODO
    @param fs: TODO
    @param vo: TODO
    @param expected_v: TODO
    @return: void

    """
    f_mat = force * np.ones((6, 6, 3))
    expected_vel_mat = expected_v * np.ones((6, 6))
    calculated_vel_mat = make_force_dep_velocity_mat(f_mat, u_vec, fs, vo)
    assert np.isclose(expected_vel_mat, calculated_vel_mat, atol=1e-6).all()

# def vhead(vo, fpar, fstall):


@pytest.mark.parametrize("s1, s2, r, a1, a2, b, ks, ho, beta, expected_val", [
    (0., 0., 0., 0., 0., -1., 1., 0., 1., 1),  # No separation
    (0., 0., 1., 0., 0., -1., 1., 1., 1., 1),  # Separation ith same rest length
    (0., 0., 3., 0., 0., -1., 1., 1., 1., exp(-2.)),
    (0., 0., 2., 0., 0., -1., 2., 1., 1., exp(-1.)),
    (1., 2., 4., 0., 0., -1., .2, 0., 1., exp(-2.5)),
    (1., 0., 1., -1., 0., 0., 2., 0., 1., exp(-4)),
    (1., 0., 1., 0., 1., 0., 2., 0., 1., exp(-2)),
])
def test_boltz_fact_mat(s1, s2, r, a1, a2, b, ks, ho, beta, expected_val):
    one_mat = np.ones((6, 6))
    S1 = s1 * one_mat
    S2 = s2 * one_mat

    expected_bf_mat = expected_val * one_mat
    calculated_bf_mat = boltz_fact_mat(S1, S2, r, a1, a2, b, ks, ho, beta)
    assert np.isclose(expected_bf_mat, calculated_bf_mat, atol=1e-8).all()

    # def make_gen_source_mat(s1_arr, s2_arr, r, a1, a2, b, ko, co, ks, ho, beta):
    # def make_gen_stretch_mat(s1, s2, u1, u2, rvec, r,):
    # def make_gen_force_mat(sgrid, s1_arr, s2_arr, u1, u2, rvec, r, ks, ho):
    # def make_gen_torque_mat(f_mat, s_arr, L, u):
