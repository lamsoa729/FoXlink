#!/usr/bin/env python
import numpy as np
from scipy import sparse
from math import exp, sqrt, cos, sin
from numba import jit, njit


"""@package docstring
File: FP_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Helper functions for FP solver.
"""


@jit
def vhead(vo, fpar, fstall):
    """!Calculate the velocity of a motor head with a smooth
    force-velocity relation

    @param vo: Unladen velocity of motor head
    @param fpar: Force on motor head parallel to rod head is attached
    @param fstall: Stall force of motor head
    @return: velocity of motor head

    """
    return vo / (1. + np.exp(-2. * (1. + (2. * fpar / fstall))))


def make_force_dep_velocity_mat(f_mat, u_vec, fs, vo):
    """!Calculate the velocity of motor heads for each point given the force at
    that point and the direction of the microtuble that head is on.

    @param f_mat: (nxnx3) matrix of force components based on head positions
    @param u_vec: unit vector of rod the motor head is on
    @param fs: stall force of motor head
    @param vo: unladen velocity of motor head
    @return: velocity matrix corresponding to position of motor head

    """
    if fs is 0:
        print("!!! Warning: motor stall force is zero. ",
              "This may cause undefined behaviour.")
    f_para_mat = np.einsum('ijk, k->ij', f_mat, u_vec)
    vel_mat = vhead(vo, f_para_mat, fs)
    return vel_mat


#######################################################################
#                    General orientation functions                    #
#######################################################################

@jit
def spring_force(s1, s2, r, a1, a2, b, ks, ho):
    """!Spring force that is generated on rod2 by rod1

    @param s1: TODO
    @param s2: TODO
    @param r: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param ks: TODO
    @param ho: TODO
    @return: TODO

    """
    return -1. * ks * (sqrt(r**2 + s1**2 + s2**2 - 2. * s1 * s2 *
                            b + 2. * r * (s2 * a2 - s1 * a1)) - ho)


@jit
def calc_alpha(s1, s2, r, a1, a2, b, ks, ho, beta):
    """!Calculate the exponent of the crosslinker's boltzmans factor

    @param s1: TODO
    @param s2: TODO
    @param r: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param k: TODO
    @param ho: TODO
    @param beta: TODO
    @return: TODO

    """
    return -.5 * beta * (spring_force(s1, s2, r, a1, a2, b, ks, ho)**2) / ks


@jit
def boltz_fact(s1, s2, r, a1, a2, b, ks, ho, beta):
    """!TODO: Calculate the boltzmann factor for a given configuration

    @param r: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param co: TODO
    @param ks: TODO
    @param ho: TODO
    @param beta: TODO
    @return: return boltzmann factor multiplied associated binding concentration

    """
    alpha = calc_alpha(s1, s2, r, a1, a2, b, ks, ho, beta)
    return exp(alpha)


# @njit
@njit(parallel=True)
def boltz_fact_mat(s1, s2, r, a1, a2, b, ks, ho, beta):
    """! Calculate the boltzmann factor for a given configuration of rods

    @param s1: Discretized position along rod 1
    @param s2: Discretized position along rod 2
    @param r: Center to center separation of rods
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param ks: TODO
    @param ho: TODO
    @param beta: TODO
    @return: boltzmann factor

    """
    bf = np.exp(-.5 * ks * beta *
                np.power((np.sqrt(r**2 + np.power(s1, 2) + np.power(s2, 2) -
                                  2. * np.multiply(s1, s2) * b +
                                  2. * r * (s2 * a2 - s1 * a1))) - ho,
                         2))
    return bf


def make_gen_source_mat(s1_arr, s2_arr, r, a1, a2, b, ko, co, ks, ho, beta):
    """! Creates a general source matrix for crosslinker attachment
    @param : TODO
    @return: TODO
    """
    S2, S1 = np.meshgrid(s2_arr, s1_arr)
    src = ko * co * boltz_fact_mat(S1, S2, r, a1, a2, b, ks, ho, beta)
    return src.round(30)  # Get rid of un-norm numbers for speed up


def make_gen_stretch_mat(s1, s2, u1, u2, rvec, r,):
    """!TODO: Docstring for make_gen_stretch_matrix.

    @param s1: TODO
    @param s2: TODO
    @param r: TODO
    @return: TODO

    """
    S2, S1 = np.meshgrid(s2, s1)
    # Create 3D array using numpy broadcasting.
    # First index is location on S1,
    # Second index is location on S2,
    # Third index is cartesian coordinate
    hvec = r * rvec + (S2[:, :, None] * u2[None, None, :] -
                       S1[:, :, None] * u1[None, None, :])
    return hvec


def make_gen_force_mat(s1_arr, s2_arr, u1, u2, rvec, r, ks, ho):
    """! Creates a general force matrix for crosslinker attachment
    @param : TODO
    @return: TODO
    """
    # Get stretch matrix (n1 x n2 x 3)
    hvec = make_gen_stretch_mat(s1_arr, s2_arr, u1, u2, rvec, r)
    if ho == 0:
        # Weight force matrix by density of crosslinkers
        f_mat = -ks * hvec
    else:
        # Get stretch matrix magnitude
        h = np.linalg.norm(hvec, axis=2)
        # Watch out for dividing by zeros
        ho_mat = np.ones(h.shape) * ho
        f_mat = -ks * (1. - np.divide(ho_mat, h, out=np.zeros_like(ho_mat),
                                      where=h != 0))
        # Weight force matrix by density of crosslinkers
        # f_mat *= sgrid
        # More vector broadcasting to give direction to force again
        f_mat = f_mat[:, :, None] * hvec[:, :, :]

    return f_mat.round(30)


def make_gen_torque_mat(f_mat, s_arr, u):
    """! Creates a general torque matrix for crosslinker attachment.
    Requires that you calculate force matrix first.
    @param : TODO
    @return: TODO
    """
    # Create vector of displacement along rod from the center of the rod
    lvec = s_arr[:, None] * u[None, :]
    # Take the cross product of all the 3 vectors of f_mat with lvec
    # TODO Test to make sure this is right
    t_mat = np.cross(lvec, f_mat)
    return t_mat


#######################################################################
#                    Angular orientation functions                    #
#######################################################################


@jit
def spring_force_ang_parallel(s1, s2, phi, ks, ho):
    """!Spring force generated on head1 parallel to rod 1.
    This is not the same for the reverse case for force on head2 parallel to rod2.
    Negative sign was sucked into second term.

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param ks: TODO
    @param ho: TODO
    @return: TODO

    """
    cosphi = cos(phi)
    return ks * (ho / sqrt(s1**2 + s2**2 - 2. * s1 *
                           s2 * cosphi) - 1) * (s1 - s2 * cosphi)


@jit
def spring_torque_ang(s1, s2, phi, ks, ho):
    """!Spring force that is generated on rod2 by rod1

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param ks: TODO
    @param ho: TODO
    @return: TODO

    """
    return (-1. * ks * s1 * s2 * sin(phi) *
            (1. - (ho / sqrt(s1**2 + s2**2 - (2. * s1 * s2 * cos(phi))))))


@jit
def calc_alpha_ang(s1, s2, phi, ks, ho, beta):
    """!Calculate the exponent of the crosslinker's boltzmans factor

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param k: TODO
    @param ho: TODO
    @param beta: TODO
    @return: TODO

    """
    return -.5 * beta * (spring_force_ang(s1, s2, phi, ks, ho)**2) / ks


@jit
def boltz_fact_ang(s1, s2, phi, ks, ho, beta):
    """!TODO: Calculate the boltzmann factor for a given configuration

    @param r: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param co: TODO
    @param ks: TODO
    @param ho: TODO
    @param beta: TODO
    @return: return boltzmann factor multiplied associated binding concentration

    """
    alpha = calc_alpha_ang(s1, s2, phi, ks, ho, beta)
    if alpha < -19.:
        return 0
    else:
        return exp(alpha)


def make_ang_source_mat(s1_arr, s2_arr, phi, ko, co, ks, ho, beta):
    """!TODO: Docstring for make_source_mat.
    @param : TODO
    @return: TODO
    """
    src = np.zeros((s1_arr.size, s2_arr.size))
    for i in range(s1_arr.size):
        for j in range(s2_arr.size):
            bf = boltz_fact_ang(s1_arr[i], s2_arr[j], phi, ks, ho, beta)
            if bf > 10e-8:
                src[i, j] = co * bf
    return sparse.csc_matrix(src)

#######################################################################
#                   Parallel orientation functions                    #
#######################################################################


def make_para_source_mat(s1_arr, s2_arr, R_pos, ko, co, ks, ho, beta):
    """!TODO: Docstring for make_para_source_mat.
    @param : TODO
    @return: TODO
    """

    src = np.zeros((s1_arr.size, s2_arr.size))
    r_vec = np.array(R_pos)

    r = np.linalg.norm(r_vec)
    a1 = r_vec[0] / r
    a2 = r_vec[0] / r
    for i in range(s1_arr.size):
        for j in range(s2_arr.size):
            bf = boltz_fact(s1_arr[i], s2_arr[j], r, a1, a2, 1., ks, ho, beta)
            if bf > 10e-8:
                src[i, j] = ko * co * bf
    return sparse.csc_matrix(src)
