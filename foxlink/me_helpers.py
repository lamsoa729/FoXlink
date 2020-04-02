#!/usr/bin/env python
"""@package docstring
File: me_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""
import numpy as np
from numba import njit


def convert_sol_to_geom(sol):
    """ Convert solution array of me_solver into 3D vectors
    @param sol: Solution numpy array greater than 11 items long

    Examples
    -------------------------
    >>> a = np.arange(18)
    >>> convert_sol_to_geom(a)
    (array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([ 9, 10, 11]))

    """
    return (sol[:3], sol[3:6], sol[6:9], sol[9:12])


def sol_print_out(sol):
    """!Print out current solution to solver

    @param r1: Center of mass postion of rod1
    @param r2: Center of mass position of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param sol: Full solution array of ODE
    @return: void

    """
    r1, r2, u1, u2 = convert_sol_to_geom(sol)
    print("Step-> r1:", r1, ", r2:", r2, ", u1:", u1, ", u2:", u2)
    print("       mu00:{}, mu10:{}, mu01:{}, mu11:{}, mu20:{}, mu02:{}".format(
        sol[12], sol[13], sol[14], sol[15], sol[16], sol[17]))
    print("       B0_j:{}, B0_i:{}, B1_j:{}, B1_i:{}, B2_j:{}, B2_i:{}".format(
        sol[18], sol[19], sol[20], sol[21], sol[22], sol[23]))


@njit
def dr_dt(f_vec, u_vec, gpara, gperp):
    """!Get the evolution of a rods postion given a force, orientation of rod,
    and drag coefficients.


    @param f_vec: Average force exerted on rod
    @param u_vec: Orientation vector of rod
    @param gpara: Parallel friction coefficient of rod
    @param gperp: Perpendicular friction coefficient of rod
    @return: Time-derivative of the rod motion

    """
    uu_mat = np.outer(u_vec, u_vec)
    # Create mobility tensor for rod
    mob_mat = np.ascontiguousarray(
        np.linalg.inv((gpara - gperp) * uu_mat + gperp * np.eye(3)))
    return np.dot(mob_mat, f_vec)


@njit
def du_dt(tau_vec, u_vec, grot):
    """!Get the evolution of a rods postion given a force, orientation of rod,
    and drag coefficients.


    @param tau_vec: Total torque exerted on rod
    @param u_vec: Orientation vector of rod
    @param grot: Rotational friction coefficient of rod
    @return: Time-derivative of the rod rotation

    """
    return np.cross(tau_vec, u_vec) / grot


@njit
def rod_geom_derivs(f_ij, tau_i, tau_j, u_i, u_j, fric_coeff):
    """!TODO: Docstring for rod_derivs.

    @param r_ij: TODO
    @param u_i: TODO
    @param u_j: TODO
    @return: TODO

    """
    (gpara_i, gperp_i, grot_i, gpara_j, gperp_j, grot_j) = fric_coeff

    # Evolution of position vectors
    dr_i = dr_dt(-1. * f_ij, u_i, gpara_i, gperp_i)
    dr_j = dr_dt(f_ij, u_j, gpara_j, gperp_j)
    # Evolution of orientation vectors
    du_i = du_dt(tau_i, u_i, grot_i)
    du_j = du_dt(tau_j, u_j, grot_j)
    return (dr_i, dr_j, du_i, du_j)
