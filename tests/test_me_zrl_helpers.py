#!/usr/bin/env python

"""@package docstring
File: test_me_zrl_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from scipy.integrate import dblquad
import foxlink.ME_zrl_helpers as meh_zrl
import foxlink.ME_gen_helpers as meh_gen
import pytest


@pytest.mark.parametrize("L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta", [
    (100, 100, 0., 0., 0., -1., 1., 1.),
    (200, 200, 0., 0., 0., -1., 1., 1.),
    (100, 100, 0., 0., 0., -1., 10., 1.),
    (100, 100, 0., 0., 0., 1., 10., 1.),
    (100, 100, 0., 0., 0., 0., 1., 1.),
])
def test_fast_zrl_src_kl_against_dblquad(
        L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta):
    for k in range(0, 2):
        for l in range(0, 2):
            fast_int = meh_zrl.fast_zrl_src_kl(
                L_i, L_j, rsqr, a_ij, a_ji, b, ks, beta, k=k, l=l)
            slow_int = dblquad(meh_gen.weighted_boltz_fact_gen, -.5 * L_i, .5 * L_i,
                               lambda s_j: -.5 * L_j, lambda s_j: .5 * L_j,
                               args=[k, l, rsqr, a_ij, a_ji, b, ks, 0., beta],
                               epsrel=1e-5)[0]
            assert fast_int == pytest.approx(slow_int, rel=1e-5)
