#!/usr/bin/env python

"""@package docstring
File: me_zrl_xl_odes.py
Author: Adam Lamson
Email: adam.r.lamson@gmail.com
Description: Class that contains the all ODEs relevant to solving the moment
expansion formalism of the Fokker-Planck equation for bound crosslinking proteins.
"""

import numpy as np


def calc_zrl_xl_moment_derivs(mu_kl, q_arr, c, ko):
    """!Calculate the time derivatives of the moments of the distribution
    function for zero rest length crosslinkers binding to moving rods.

    @param mu_kl: Moments of the distribution function
    @param q_arr: Geometric factors for binding
    @param ko: On rate of crosslinkers
    @return: Time derivatives of unbound crosslinkers and moments
    """

    dmu_kl = np.array([ko * (c * q - mu) for q, mu in zip(q_arr, mu_kl)])

    return dmu_kl


def calc_zrl_unbound_and_xl_moment_derivs(n_unbound, mu_kl, q_arr, ko):
    dn_unbound = ko * (mu_kl[0] - q_arr[0] * n_unbound)

    dmu_kl = np.array([ko * (n_unbound * q - mu) for q, mu in zip(q_arr, mu_kl)])

    return dn_unbound, dmu_kl
