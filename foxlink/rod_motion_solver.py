#!/usr/bin/env python
import numpy as np
from numba import njit
from .solver import Solver
from .rod_steric_forces import calc_wca_force_torque


"""@package docstring
File: rod_motion_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@njit
def get_rod_drag_coeff(visc, L, d):
    """! Get the drag coeffiecents of slender bodies using theory from
    Brownian dynamics of hard spherocylinders
    Hartmut Löwen
    Phys. Rev. E 50, 1232 – Published 1 August 1994

    @param visc: Viscosity of fluid
    @param L: Length of rod
    @param d: Diameter of rod
    @return: parallel, perpendicular, and rotational drag coefficients

    """
    l = L / d
    ln_l = np.log(l)
    l2 = l * l
    g_para = ((2. * np.pi * visc * L * l2) /
              (l2 * (ln_l - .207) + (l * .98) - .133))
    g_perp = ((4. * np.pi * visc * L * l2) /
              (l2 * (ln_l + .839) + (0.185 * l) + 0.233))
    g_rot = ((np.pi * visc * L * L * L * l2) /
             (3. * (l2 * (ln_l - .662) + (.917 * l) - .05)))

    return g_para, g_perp, g_rot


def get_rod_mob_mat(visc, L, d, R_vec):
    """! Create and return 3x3 mobility matrix for rod

    @param visc: Viscosity of fluid
    @param L: Length of rod
    @param d: Diameter of rod
    @param R_vec: Orientation of rod
    @return: Mobility matrix, rotaional drag coefficient

    """
    # Calculate the diagnol elements of linear drag tensor and
    #   rotation drag coefficient
    g_para, g_perp, g_rot = get_rod_drag_coeff(visc, L, d)
    # Create dyadic tensor of orientation vector
    uu_mat = np.outer(R_vec, R_vec)
    # Create mobility tensor for rod
    mob_mat = np.linalg.inv((g_para - g_perp) * uu_mat + g_perp * np.eye(3))
    return mob_mat, g_rot


class RodMotionSolver(Solver):

    """!Docstring for RodMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of system
        Note: parameter file needs viscosity in order to run

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init RodMotionSolver ->", end=" ")
        Solver.__init__(self, pfile=pfile, pdict=pdict)

    def RodStep(self, force1=0, force2=0, torque1=0, torque2=0,
                r_i=None, r_j=None, u_i=None, u_j=None):
        """! Change the position of rods based on forces and torques exerted on rod
        @param force: Force vector of rod2 by rod1
        @param torque: Torque vector of rod2 by rod1
        @param r_i: TODO
        @param r_j: TODO
        @param u_i: TODO
        @param u_j: TODO
        @return: new positions and orientations of rods

        """
        # Calculate drag coefficients of rods:
        #   Requires viscocity of liquid, rod diameters, and rod lengths
        visc = self._params["viscosity"]
        L_i = self._params["L1"]
        L_j = self._params["L2"]
        d = self._params["rod_diameter"]

        self.calc_rod_steric_interactions(r_i, r_j, u_i, u_j,
                                          L_i, L_j, d)

        f_i = force1 + self.steric_force_i
        f_j = force2 + self.steric_force_j
        tau_i = torque1 + self.steric_torque_i
        tau_j = torque2 + self.steric_torque_j

        if self.steric_flag == 'constrained':
            # TODO: Work on this tomorrow <02-04-20, ARL> #
            pass

        if (np.any(f_i) or np.any(f_j) or np.any(tau_i) or np.any(tau_j)):
            # Get the mobility matrices and rotational drag coefficient
            mob_mat1, g_rot1 = get_rod_mob_mat(visc, L_i, d, u_i)
            mob_mat2, g_rot2 = get_rod_mob_mat(visc, L_j, d, u_j)
        # Use forces and torques to evolve system using a Forward Euler scheme
        if np.any(f_i):
            r_i += self.dt * np.dot(mob_mat1, f_i)
        if np.any(f_j):
            r_j += self.dt * np.dot(mob_mat2, f_j)
        if np.any(tau_i):
            u_i += np.cross(tau_i, u_i) * self.dt / g_rot1
            u_i /= np.linalg.norm(u_i)  # Renormalize just in case
        if np.any(tau_j):
            u_j += np.cross(tau_j, u_j) * self.dt / g_rot2
            u_j /= np.linalg.norm(u_j)  # Renormalize just in case

        # Recalculate source matrix after rods have moved
        self.calcSourceMatrix()
        return r_i, r_j, u_i, u_j
