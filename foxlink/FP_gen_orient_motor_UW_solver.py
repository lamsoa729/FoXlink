#!/usr/bin/env python
from .FP_motor_UW_solver import FPMotorUWSolver
from .FP_gen_orient_solver import FPGenOrientSolver
from .FP_helpers import make_force_dep_velocity_mat


"""@package docstring
File: FP_gen_orient_motor_UW.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenOrientMotorUWSolver(FPMotorUWSolver, FPGenOrientSolver):

    """!    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        makeSourceMat: FPGenOrientSolver
        makeForceMat: FPGenOrientSolver
        makeTorueMat: FPGenOrientSolver
        makeDiagMats: FPUWMotorSolver
        Step: FPUWMotorSolver
        RodStep: None


        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPGenOrientMotorUWSolver ->", end=" ")
        FPGenOrientSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()
        # Since the rods do not move you only need to do this once
        self.calcVelocityMats()

    def calcVelocityMats(self):
        """!Calculate the motor head velocities as a function of position on each rod.
        @return: TODO

        """
        vo = self._params['vo']
        fs = self._params['fs']
        # Velocity of heads on the first rod. Negative because force matrix is
        # the force that the second head experiences.
        self.vel_mat1 = make_force_dep_velocity_mat(
            -1. * self.f_mat, self.R1_vec, fs, vo)
        # Velocity of the heads on the second rod
        self.vel_mat2 = make_force_dep_velocity_mat(
            self.f_mat, self.R2_vec, fs, vo)
