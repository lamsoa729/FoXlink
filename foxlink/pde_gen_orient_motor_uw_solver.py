#!/usr/bin/env python
from .pde_motor_uw_solver import PDEMotorUWSolver
from .pde_gen_orient_solver import PDEGenOrientSolver
from .pde_helpers import make_force_dep_velocity_mat
import numpy as np


"""@package docstring
File: pde_gen_orient_motor_UW.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEGenOrientMotorUWSolver(PDEMotorUWSolver, PDEGenOrientSolver):

    """!    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: PDEGenOrientSolver
        makeSourceMat: PDEGenOrientSolver
        makeForceMat: PDEGenOrientSolver
        makeTorueMat: PDEGenOrientSolver
        makeDiagMats: PDEUWSolver
        Step: PDEUWMotorSolver
        RodStep: None

        @param pfile: parameter file path
        @param pdict: parameter dictionary

        """
        print("Init PDEGenOrientMotorUWSolver ->", end=" ")
        PDEGenOrientSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()
        # Since the rods do not move you only need to do this once
        self.calcVelocityMats()

    def calcVelocityMats(self):
        """!Calculate the motor head velocities as a function of position on each rod.
        @return: void, calculates velocity matrices self.vel_mat1, self.vel_mat2

        """
        vo = self._params['vo']
        fs = self._params['fs']
        # Velocity of heads on the first rod. Negative because force matrix
        # represents the force that the second heads experiences.
        self.vel_mat1 = make_force_dep_velocity_mat(
            -1. * self.f_mat, self.R1_vec, fs, vo)
        # Velocity of the heads on the second rod
        self.vel_mat2 = make_force_dep_velocity_mat(
            self.f_mat, self.R2_vec, fs, vo)
