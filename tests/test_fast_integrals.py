#!/usr/bin/env python

"""@package docstring
File: test_me_zrl_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from scipy.integrate import dblquad
import foxlink.create_me_1D_evolver as cmde
import pytest


@pytest.mark.parametrize("L, x, ui, uj, ks, beta", [
    (100., 0., 1., 1., 1., 1.),
    (100., 0., 1., -1., 1., 1.),
    (100., 50., 1., -1., 1., 1.),
    (100., -50., 1., -1., 1., 1.),
    (100., 50., 1., 1., 1., 1.),
    (100., 0., 1., -1., 1., 1.),
    (100., 0., -1., -1., 1., 1.),
    (200., 0., 1., -1., 1., 1.),
    (100., 0., 1., -1., 10., 1.),
    (100., 0., 1., -1., .5, 1.),
    (100., 0., 1., 1., .5, 1.),
    (10., 0., 1., -1., .5, 1.),
    (10., 5., 1., -1., .5, 1.),
])
def test_fast_1D_src_kl_against_dblquad(
        L, x, ui, uj, ks, beta):
    for k in range(0, 2):
        for l in range(0, 2):
            print("k: {}, l: {}".format(k, l))
            fast_int = cmde.fast_1D_src_kl(
                L, x, ui, uj, ks, beta, k=k, l=l)
            L_temp = L
            slow_int = dblquad(cmde.weighted_boltz_fact_1D, -.5 * L, .5 * L,
                               lambda sj: -.5 * L_temp, lambda sj: .5 * L_temp,
                               args=[x, ui, uj, ks, beta, k, l],
                               epsrel=1e-5)[0]
            assert fast_int == pytest.approx(slow_int, rel=1e-5)
