#!/usr/bin/env python
from .FP_gen_orient_motor_UW_solver import FPGenOrientMotorUWSolver
from .rod_motion_solver import RodMotionSolver


"""@package docstring
File: FP_gen_motion_motor_UW.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionMotorUWSolver(RodMotionSolver, FPGenOrientMotorUWSolver):

    """!    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: FPGenOrientSolver
        calcTorqueMat: FPGenOrientSolver
        calcVelocityMats: FPGenOrientMotorUWSolver
        makeDiagMats: FPUWMotorSolver
        stepUW: FPUWSolver
        Step: self
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
            self.force1, self.force2, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
        # Update
        self.calcVelocityMats()
