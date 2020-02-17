#!/usr/bin/env python
"""@package docstring
File: pde_gen_motion_motor_uw_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""
from .pde_gen_orient_motor_uw_solver import PDEGenOrientMotorUWSolver
from .rod_motion_solver import RodMotionSolver


class PDEGenMotionMotorUWSolver(RodMotionSolver, PDEGenOrientMotorUWSolver):

    """!Docstring for PDEGenMotionMotorUWSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: PDEGenOrientSolver
        calcSourceMatrix: PDEGenOrientSolver
        calcForceMatrix: PDEGenOrientSolver
        calcTorqueMatrix: PDEGenOrientSolver
        calcVelocityMats: PDEGenOrientMotorUWSolver
        makeDiagMats: PDEUWSolver
        stepUW: PDEUWSolver
        Step: self
        RodStep: PDEGenMotionSolver

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init PDEGenMotionMotorUWSolver ->", end=" ")
        PDEGenOrientMotorUWSolver.__init__(self, pfile, pdict)

    def Step(self):
        """!Step both motor heads and rods in time
        @return: TODO

        """
        # Update xlink positions
        PDEGenOrientMotorUWSolver.Step(self)
        # Calculate new forces and torque
        self.calcInteractions()

        # Update rod positions and recalculate source matrices
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force1, self.force2, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
        # Update velocity matrices for next time
        self.calcVelocityMats()
