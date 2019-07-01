#!/usr/bin/env python
import numpy as np
from numba import njit
from .solver import Solver


"""@package docstring
File: FP_rod_motion_solver.py
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

    @param arg1: TODO
    @return: TODO

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

    @param visc: TODO
    @param L: TODO
    @param d: TODO
    @param R_vec: TODO
    @return: TODO

    """
    # Calculate the diagnol elements of linear drag tensor and
    #   rotation drag coefficient
    g_para, g_perp, g_rot = get_rod_drag_coeff(visc, L, d)
    # Create dyadic tensor of orientation vector
    uu_mat = np.outer(R_vec, R_vec)
    # Create mobility tensor for rod
    mob_mat = np.linalg.inv((g_para - g_perp) * uu_mat + g_perp * np.eye(3))
    return mob_mat, g_rot


class FPRodMotionSolver(Solver):

    """!Docstring for FPRodMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system
        Note: parameter file needs viscosity in order to run

        @param pfile: TODO
        @param name: TODO

        """
        print("Init FPRodMotionSolver ->", end=" ")
        Solver.__init__(self, pfile=pfile, pdict=pdict)

    def RodStep(self, force=0, torque1=0, torque2=0, R1_pos=None,
                R2_pos=None, R1_vec=None, R2_vec=None):
        """! Change the position of rods based on forces and torques exerted on rod
        @param force: Force vector of rod2 by rod1
        @param torque: Torque vector of rod2 by rod1
        @param R1_pos: TODO
        @param R2_pos: TODO
        @param R1_vec: TODO
        @param R2_vec: TODO
        @return: void

        """
        # Calculate drag coefficients of rods:
        #   Requires viscocity of liquid, rod diameters, and rod lengths
        visc = self._params["viscosity"]
        L1 = self._params["L1"]
        L2 = self._params["L2"]
        d = self._params["rod_diameter"]
        if force is not 0 and torque1 is not 0 and torque2 is not 0:
            # Get the mobility matrices and rotational drag coefficient
            mob_mat1, g_rot1 = get_rod_mob_mat(visc, L1, d, R1_vec)
            mob_mat2, g_rot2 = get_rod_mob_mat(visc, L2, d, R2_vec)
        # Use forces and torques to evolve system using a Forward Euler scheme
        if force is not 0:
            R2_pos += self.dt * np.dot(mob_mat2, force)
            R1_pos -= self.dt * np.dot(mob_mat1, force)
        if torque1 is not 0:
            R1_vec += np.cross(torque1, R1_vec) * self.dt / g_rot1
            # Renormalize orientation vectors
            R1_vec /= np.linalg.norm(R1_vec)
        if torque2 is not 0:
            R2_vec += np.cross(torque2, R2_vec) * self.dt / g_rot2
            R2_vec /= np.linalg.norm(R2_vec)
        # Recalculate source matrix after rods have moved
        self.calcSourceMatrix()
        return R1_pos, R2_pos, R1_vec, R2_vec


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
