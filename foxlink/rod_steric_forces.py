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
    sigma6 = sigma**6

    rcut = np.power(2.0, 1.0 / 6.0) * sigma

    # Impose force only if LJ potential is repulsive
    f_mag = (24. * eps * sigma6 * r_inv6 * r_inv) * (
        2. * sigma6 * r_inv6 - 1.) if r_mag < rcut else 0.
    return f_mag * u_vec


def find_sphero_min_dist(r_i, r_j, u_i, u_j, L_i, L_j):
    """!Find the minimum distance between two spherocylinders (i,j) and the
    points on the spherocylinders where that minimum distance occurs. Minimum
    distance points are relative to the center of the spherocylinders.

    @param r_i: Center of rod i
    @param r_j: Center of rod j
    @param u_i: Orientation vector of rod i
    @param u_j: Orientation vector of rod j
    @param L_i: Length of rod i
    @param L_j: Length of rod j
    @return: Minimum distance vector pointing from i to j,
    point of minimum distance on rod i, point of minium distance rod j

    """
    min_vec_ij = 0
    l_i = 0
    l_j = 0

    return min_vec_ij, l_i, l_j


def calc_wca_force_torque(r_i, r_j, u_i, u_j, L_i, L_j, rod_diameter, eps):
    """!Calculate and return the forces and torques on two spherocylinders (i,j).

    @param r_i: TODO
    @param r_j: TODO
    @param u_i: TODO
    @param u_j: TODO
    @param L_i: TODO
    @param L_j: TODO
    @param rod_diameter: TODO
    @param eps: TODO
    @return: TODO

    """
    dr_ij, l_i, l_j = find_sphero_min_dist(r_i, r_j, u_i, u_j, L_i, L_j)

    f_ij = wca_force(dr_ij, rod_diameter, eps)
    tau_i = np.cross(l_i * u_i, -dr_ij)
    tau_j = np.cross(l_j * u_j, dr_ij)

    return f_ij, tau_i, tau_j
