#!/usr/bin/env python
from .optical_trap_motion_solver import OpticalTrapMotionSolver
from .pde_gen_motion_motor_uw_solver import PDEGenMotionMotorUWSolver


"""@package docstring
File: pde_ot_gen_motion_motor_uw_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEOpticalTrapGenMotionMotorUWSolver(
        OpticalTrapMotionSolver, PDEGenMotionMotorUWSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: PDEGenOrientSolver
        calcSourceMat: PDEGenOrientSolver
        calcForceMat: PDEGenOrientSolver
        calcTorqueMat: PDEGenOrientSolver
        calcVelocityMats: PDEGenOrientMotorUWSolver
        makeDiagMats: PDEUWMotorSolver
        stepUW: PDEUWSolver
        Step: PDEGenMotionMotorUWSolver
        RodStep: OpticalTrapMotionSolver
        Write: self
        makeDataframe: self

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init PDEOpticalTrapGenMotionMotorUWSolver ->", end=" ")
        PDEGenMotionMotorUWSolver.__init__(self, pfile, pdict)
        self.OTParseParams()
        self.calcOTInteractions(self.R1_pos,
                                self.R2_pos,
                                self.R1_vec,
                                self.R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: void, Create dataframe for output with optical trap data

        """
        PDEGenMotionMotorUWSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        i_step = PDEGenMotionMotorUWSolver.Write(self)
        self.OTWrite(i_step)
        return i_step
