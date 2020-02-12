#!/usr/bin/env python

"""@package docstring
File: rod_steric_forces.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from numba import jit, njit


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


@jit
def match_sign(a, b):
    """!TODO: Docstring for match_sign.

    @param a: TODO
    @param b: TODO
    @return: TODO

    """
    return abs(a) if b >= 0. else -abs(a)


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
    # Create scalar values for computational ease
    r_ij = r_j - r_i
    a_ij = np.dot(u_i, r_ij)
    a_ji = np.dot(u_j, -1. * r_ij)
    b_ij = np.dot(u_i, u_j)
    hL_i = .5 * L_i
    hL_j = .5 * L_j
    denom = 1.0 - (b_ij**2)

    # Compute minimum distance (see Allen et al., Adv. Chem. Phys. 86, 1
    # (1993)). First consider two infinitely long lines.
    if denom < 1.0E-8:
        l_i = .5 * a_ij
        l_j = .5 * a_ji
    else:
        l_i = (a_ij + b_ij * a_ji) / denom
        l_j = (a_ji + b_ij * a_ij) / denom
    l_i_mag = abs(l_i)
    l_j_mag = abs(l_j)

    # Now take into account the fact that the line segments are finite length.
    # If the infinite lines are closest at points beyond the rod line segments.
    if l_i_mag > hL_i and l_j_mag > hL_j:

        # Look at the end of rod i first. Consider it case "a".
        l_i_a = match_sign(hL_i, l_i)
        l_j_a = a_ji + l_i_a * b_ij
        l_j_mag = abs(l_j_a)
        if l_j_mag > hL_j:
            l_j_a = match_sign(hL_j, l_j_a)

        min_vec_ij_a = r_ij - (l_i_a * u_i) + (l_j_a * u_j)
        min_vec_ij_mag2_a = np.dot(min_vec_ij_a, min_vec_ij_a)

        # Look at the end of rod j. Consider it case "b".
        l_j_b = match_sign(hL_j, l_j)
        l_i_b = a_ij + l_j_b * b_ij
        l_i_mag = abs(l_i_b)
        if l_i_mag > hL_i:
            l_i_b = match_sign(hL_i, l_i_b)

        # Calculate minimum distance between two spherocylinders.
        min_vec_ij_b = r_ij - (l_i_b * u_i) + (l_j_b * u_j)
        min_vec_ij_mag2_b = np.dot(min_vec_ij_b, min_vec_ij_b)

        # Choose the smallest minimum distance.
        if min_vec_ij_mag2_a < min_vec_ij_mag2_b:
            return min_vec_ij_a, l_i_a, l_j_a
        return min_vec_ij_b, l_i_b, l_j_b

    # If we know that only l_i is larger than its rod length
    if l_i_mag > hL_i:

        # Adjust l_i and l_j since l_i must be shifted to end of rod i
        l_i = match_sign(hL_i, l_i)
        l_j = a_ji + l_i * b_ij
        l_j_mag = abs(l_j)
        if l_j_mag > hL_j:
            l_j = match_sign(hL_j, l_j)

        min_vec_ij = r_ij - (l_i * u_i) + (l_j * u_j)
        return min_vec_ij, l_i, l_j

    # If we know that only l_j is larger than its rod length
    if l_j_mag > hL_j:

        # Adjust l_i and l_j since l_i must be shifted to end or rod i
        l_j = match_sign(hL_j, l_j)
        l_i = a_ij + l_j * b_ij
        l_i_mag = abs(l_i)
        if l_i_mag > hL_i:
            l_i = match_sign(hL_i, l_i)

        min_vec_ij = r_ij - (l_i * u_i) + (l_j * u_j)
        return l_i, l_j, min_vec_ij

    # If neither l_i nor l_j is larger than there rods length just calculate
    # minimum distance using l_i and l_j
    min_vec_ij = r_ij - (l_i * u_i) + (l_j * u_j)
    # min_vec_ij_mag2 = np.dot(min_vec_ij, min_vec_ij)

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
    min_vec_ij, l_i, l_j = find_sphero_min_dist(r_i, r_j, u_i, u_j, L_i, L_j)

    f_ij = wca_force(min_vec_ij, rod_diameter, eps)
    tau_i = np.cross(l_i * u_i, -min_vec_ij)
    tau_j = np.cross(l_j * u_j, min_vec_ij)

    return f_ij, tau_i, tau_j
