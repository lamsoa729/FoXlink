#!/usr/bin/env python
import numpy as np
from scipy import sparse
from math import exp, sqrt, cos, sin
from numba import jit, njit


"""@package docstring
File: pde_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Helper functions for PDE solver.
"""


@njit
def vhead_smooth(vo, fpar, fstall):
    """!Calculate the velocity of a motor head with a smooth
    force-velocity relation

    @param vo: Unladen velocity of motor head
    @param fpar: Force on motor head parallel to rod head is attached
    @param fstall: Stall force of motor head
    @return: velocity of motor head

    """
    return vo / (1. + np.exp(-2. * (1. + (2. * fpar / fstall))))


def vhead(vo, fpar, fstall):
    """!Calculate the velocity of a motor head with a linear force-velocity
    relation.

    @param vo: Unladen velocity of motor head
    @param fpar: Force on motor head parallel to rod head is attached
    @param fstall: Stall force of motor head
    @return: velocity of motor head

    """
    # Clip sets bounds between 0, 1 necessary for linear force-velocity
    # relation
    return vo * np.clip(1. + (fpar / fstall), 0., 1.)


def make_force_dep_velocity_mat(f_mat, u_vec, fs, vo):
    """!Calculate the velocity of motor heads for each point given the force at
    that point and the direction of the microtuble that head is on.

    @param f_mat: (nxnx3) matrix of force components based on head positions
    @param u_vec: unit vector of rod the motor head is on
    @param fs: stall force of motor head
    @param vo: unladen velocity of motor head
    @return: velocity matrix corresponding to position of motor head

    """
    if fs == 0:
        print("!!! Warning: motor stall force is zero. ",
              "This may cause undefined behaviour.")
    f_para_mat = np.einsum('ijk, k->ij', f_mat, u_vec)
    vel_mat = vhead(vo, f_para_mat, fs)
    return vel_mat


#######################################################################
#                    General orientation functions                    #
#######################################################################

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
    s2, s1 = np.meshgrid(s2_arr, s1_arr)
    src = ko * co * boltz_fact_mat(s1, s2, r, a1, a2, b, ks, ho, beta)
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
