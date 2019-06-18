#!/usr/bin/env python
import numpy as np
from scipy import sparse
from .solver import Solver


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def C11Neumann(t, s1, s2, L1, L2, gamma, beta, C0=1., C11=1.0):
    """! First eigenmode of the 2D diffusion equation with Neumann boundary conditions

    @param t: Time
    @param s1: Distance in the first dimentsion
    @param s2: Distance in the second dimension
    @param L1: Length between boundaries in the first dimension
    @param L2: Length between boundaries in the second dimension
    @param gamma: The friction coefficient (related to the diffusion coeffiecient)
    @param beta: Inverse temperature
    @return: Value of the solution at a specific time and place

    """
    D = 1. / (gamma * beta)
    a_m = np.pi / L1
    a_n = np.pi / L2
    mu = D * (a_m**2 + a_n**2)
    return C0 + (C11 * np.exp(-mu * t) * np.cos(a_m * s1) * np.cos(a_n * s2))


def C11NeumannInit(solver, C0=1., C11=1.0):
    """!Initialize solver with the first eigenmode of the 2D diffusion equation

        @return: sparse matrix of initial state of eigen funcition

    """
    s1_arr = solver.s1
    s2_arr = solver.s2
    params = solver._params
    L1 = params['L1']
    L2 = params['L2']
    gamma = params['gamma']
    beta = params['beta']
    xl_dens = np.zeros((s1_arr.size, s2_arr.size))
    for i in range(s1_arr.size):
        for j in range(s2_arr.size):
            xl_dens[i, j] = C11Neumann(
                0, s1_arr[i], s2_arr[j], L1, L2, gamma, beta, C0, C11)
    return sparse.csc_matrix(xl_dens)


def CmnNeumann(t, s1, s2, L1, L2, gamma, beta, m=1, n=1, C0=1., Cmn=1.0):
    """! N,M eigenmode of the 2D diffusion equation with Neumann boundary conditions

    @param t: Time
    @param s1: Distance in the first dimentsion
    @param s2: Distance in the second dimension
    @param L1: Length between boundaries in the first dimension
    @param L2: Length between boundaries in the second dimension
    @param gamma: The friction coefficient (related to the diffusion coeffiecient)
    @param beta: Inverse temperature
    @param m: mth eigenvalue along the s1 dimension
    @param n: nth eigenvalue along the s2 dimension
    @param C0: Constant in front of solution
    @param Cmn: Constant in front of the N,M eigen function
    @return: Value of the solution at a specific time and place

    """
    D = 1. / (gamma * beta)
    a_m = m * np.pi / L1
    a_n = n * np.pi / L2
    mu = D * (a_m**2 + a_n**2)
    return C0 + (Cmn * np.exp(-mu * t) * np.cos(a_m * s1) * np.cos(a_n * s2))


def CmnNeumannInit(solver, m=1, n=1, C0=1., Cmn=1.0):
    """!Initialize solver with the first eigenmode of the 2D diffusion equation
    @param solver: Solver object initalized with parameters
    @param m: mth eigenfunction along the s1 dimension
    @param n: nth eigenfunction along the s2 dimension
    @param C0: Constant in front of solution
    @param Cmn: Constant in front of the N,M eigen function

        @return: sparse matrix of initial state of eigen funcition

    """
    # Get dimensional arrays
    s1_arr = solver.s1
    s2_arr = solver.s2
    # Get system parameters from solver
    params = solver._params
    L1 = params['L1']
    L2 = params['L2']
    gamma = params['gamma']
    beta = params['beta']
    xl_dens = np.zeros((s1_arr.size, s2_arr.size))
    for i in range(s1_arr.size):
        for j in range(s2_arr.size):
            xl_dens[i, j] = CmnNeumann(
                0, s1_arr[i], s2_arr[j], L1, L2, gamma, beta, m, n, C0, Cmn)
    return sparse.csc_matrix(xl_dens)


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
