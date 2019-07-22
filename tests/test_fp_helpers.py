#!/usr/bin/env python
import pytest
import numpy as np
from foxlink.FP_helpers import make_force_dep_velocity_mat


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
# def boltz_fact_mat(s1, s2, r, a1, a2, b, ks, ho, beta):
# def make_gen_source_mat(s1_arr, s2_arr, r, a1, a2, b, ko, co, ks, ho, beta):
# def make_gen_stretch_mat(s1, s2, u1, u2, rvec, r,):
# def make_gen_force_mat(sgrid, s1_arr, s2_arr, u1, u2, rvec, r, ks, ho):
# def make_gen_torque_mat(f_mat, s_arr, L, u):
