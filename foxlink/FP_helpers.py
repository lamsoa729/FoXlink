#!/usr/bin/env python
import numpy as np
from scipy import sparse
from math import exp, sqrt, cos, sin
from numba import jit


"""@package docstring
File: FP_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Helper functions for FP solver
"""


@jit
def spring_force_ang(s1, s2, phi, ks, ho):
    """!Spring force that is generated on head 1 by head 2 or vice versa.
    Whatch the negative sign

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param ks: TODO
    @param ho: TODO
    @return: TODO

    """
    return -1. * ks * (sqrt(s1**2 + s2**2 - 2. * s1 * s2 * cos(phi)) - ho)


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


def boltz_fact_mat(s1, s2, r, a1, a2, b, ks, ho, beta):
    """! Calculate the boltzmann factor for a given configuration

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
    bf = np.exp(-.5 * ks * np.power((np.sqrt(r**2 +
                                             np.power(s1, 2) +
                                             np.power(s2, 2) -
                                             2. * np.multiply(s1, s2) * b +
                                             2. * r * (s2 * a2 - s1 * a1)) - ho),
                                    2))

    return bf


@jit
def vhead(vo, fpar, fstall):
    """!Calculate the velocity of a motor head with a smooth
    force-velocity relation

    @param vo: TODO
    @param fpar: TODO
    @param fstall: TODO
    @return: velocity of motor head

    """
    return vo / (1. + exp(-2. * (1. + (2. * fpar / fstall))))


@jit
def laplace_5p(i, j, sgrid, ds):
    """!Find the laplacian using the 4-point method

    @param i: TODO
    @param j: TODO
    @param sol: TODO
    @return: TODO

    """
    return (sgrid[i - 1, j] + sgrid[i + 1, j] + sgrid[i, j - 1] +
            sgrid[i, j + 1] - (4. * sgrid[i, j])) / (ds * ds)


def make_solution_grid(lim1, lim2, ds):
    """!TODO: Docstring for gen_solution_grid.

    @param lim1: TODO
    @param lim2: TODO
    @param ds: TODO
    @return: TODO

    """
    ns1 = int(lim1 / ds) + 2
    ns2 = int(lim2 / ds) + 2

    # Discrete rod locations
    s1 = np.linspace(0, lim1 - ds, self.ns1) + (ds * .5)
    s2 = np.linspace(0, lim2 - ds, self.ns2) + (ds * .5)

    # Solution grids
    sgrid = sparse.csc_matrix((self.ns1, self.ns2))

    return sgrid, ns1, ns2, s1, s2


def make_ang_source_mat(s1_arr, s2_arr, phi, co, ks, ho, beta):
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


def make_para_source_mat(s1_arr, s2_arr, R_pos, co, ks, ho, beta):
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
                src[i, j] = co * bf
    return sparse.csc_matrix(src)


#######################################################################
#                    General orientation functions                    #
#######################################################################


def make_gen_source_mat(s1_arr, s2_arr, r, a1, a2, b, ko, co, ks, ho, beta):
    """! Creates a general source matrix for crosslinker attachment
    @param : TODO
    @return: TODO
    """
    S2, S1 = np.meshgrid(s2_arr, s1_arr)
    src = ko * co * boltz_fact_mat(S1, S2, r, a1, a2, b, ks, ho, beta)
    # return sparse.csc_matrix(src)
    return src


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


def make_gen_force_mat(sgrid, s1_arr, s2_arr, u1,
                       u2, rvec, r, ks, ho):
    """! Creates a general force matrix for crosslinker attachment
    @param : TODO
    @return: TODO
    """
    # Get stretch matrix (n1 x n2 x 3)
    hvec = make_gen_stretch_mat(s1_arr, s2_arr, u1, u2, rvec, r)
    if ho == 0:
        # Weight force matrix by density of crosslinkers
        f_mat = -ks * sgrid
        f_mat = f_mat[:, :, None] * hvec[:, :, :]
    else:
        # Get stretch matrix magnitude
        h = np.linalg.norm(hvec, axis=2)
        # Watch out for dividing by zeros
        ho_mat = np.ones(h.shape) * ho
        f_mat = -ks * (1. - np.divide(ho_mat, h, out=np.zeros_like(ho_mat),
                                      where=h != 0))
        # Weight force matrix by density of crosslinkers
        f_mat *= sgrid
        # More vector broadcasting to give direction to force again
        f_mat = f_mat[:, :, None] * hvec[:, :, :]

    return f_mat


def make_gen_torque_mat(f_mat, s_arr, L, u):
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


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
