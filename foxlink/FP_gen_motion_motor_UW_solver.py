#!/usr/bin/env python
from .FP_gen_motion_motor_UW_solver import FPGenOrientMotorUWSolver
from .FP_rod_motion_solver import FPRodMotionSolver
from .FP_helpers import make_force_dep_velocity_mat


"""@package docstring
File: FP_gen_orient_motor_UW.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionMotorUWSolver(FPRodMotionSolver, FPGenOrientMotorUWSolver):

    """!    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: FPGenOrientSolver
        calcTorueMat: FPGenOrientSolver
        calcVelocityMats: FPGenOrientMotorUWSolver
        makeDiagMats: FPUWMotorSolver
        stepUW: FPUWSolver
        Step: FPUWMotorSolver
        RodStep: FPGenMotionSolver


        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPGenMotionMotorUWSolver ->", end=" ")
        FPGenOrientMotorUWSolver.__init__(self, pfile, pdict)

    def Step(self):
        """!Step both motor heads and rods in time
        @return: TODO

        """
        # Update xlink positions
        FPGenOrientMotorUWSolver.Step(self)
        # Calculate new forces and torque
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions and recalculate source matrices
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force, self.torque, self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
        # Update
        self.calcVelocityMats()
