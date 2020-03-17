#!/usr/bin/env python

"""@package docstring
File: test_bivariate_gauss_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from math import erf
import numpy as np
from scipy.integrate import quad, dblquad
import foxlink.bivariate_gauss_helpers as bgh
import pytest


# @pytest.mark.parametrize("L_i, L_j, mu00, mu10, mu01, mu11, mu20, mu02", [
#     (10., 10., 1., 0., 0., 0., 1., 1.),
#     (1., 1., 1., 0., 0., 0., 1., 1.),
#     # (10., 10., 1., 0.5, 0.5, .25, 1., 1.),
#     # (100., 100., 1., 0.5, 0.5, .25, 1., 1.),
#     # FIXME If nu is not zero, this integration method is not accurate
#     # (100., 100., 1., 0.4, 0.4, .25, 1., 1.),
#     # (1., 1., 1., 0.4, 0.4, .25, 1., 1.),
#     # (1., 1., 1., 0.0, 0.0, 0, 1., 1.),
#     # (1., 1., 1., 0., 0., .1, 1., 1.),
#     # (1., 1., 1., 0., 0., .5, 1., 1.),
# ])
# def test_fast_gauss_moment_kl(
#         L_i, L_j, mu00, mu10, mu01, mu11, mu20, mu02):
#     for k in range(0, 2):
#         for l in range(0, 2):
#             (mu10_bar, mu01_bar,
#              sigma_i, sigma_j,
#              nu, gamma) = bgh.convert_moments_to_gauss_vars(mu00, mu10, mu01,
#                                                             mu11, mu20, mu02)

#             fast_int = bgh.fast_gauss_moment_kl(
#                 L_i, L_j, mu00, mu10, mu01, mu11, mu20, mu02, k=k, l=l)
#             slow_int = dblquad(bgh.weighted_bivariate_gauss, -.5 * L_i, .5 * L_i,
#                                lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
#                                args=[mu00, mu10_bar, mu01_bar, sigma_i, sigma_j,
#                                      nu, gamma, k, l],
#                                epsrel=1e-12)[0]
#             print("k = {}, l = {}".format(k, l))
#             assert fast_int == pytest.approx(slow_int, rel=1e-4)
