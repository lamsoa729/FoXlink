#!/usr/bin/env python

"""@package docstring
File: rod_steric_forces.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from numba import jit


@jit
def wca_force(dr, sigma, eps):
    """!Calculate the magnitude of the WCA force between points

    @param dr: Vector between from point i to point j
    @param sigma: Diameter of points
    @param eps: Energy scale of interaction
    @return: Force vector on point j from i

    """
    r_mag = np.linalg.norm(dr)
    r_inv = 1. / r_mag
    u_vec = dr * r_inv
    r_inv6 = r_inv**6

    rcut = np.power(2.0, 1.0 / 6.0) * sigma
    sigma6 = sigma**6

    fmag = 24. * eps * sigma6 * (r_inv6 * r_inv) * (2. * sigma6 * r_inv6 -
                                                    1.) if r_mag < rcut else 0.
    return fmag * u_vec
