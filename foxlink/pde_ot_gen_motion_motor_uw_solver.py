#!/usr/bin/env python
from .optical_trap_motion_solver import OpticalTrapMotionSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver


"""@package docstring
File: FP_OT_gen_motion_motor_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPOpticalTrapGenMotionMotorUWSolver(
        OpticalTrapMotionSolver, FPGenMotionMotorUWSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: FPGenOrientSolver
        calcTorqueMat: FPGenOrientSolver
        calcVelocityMats: FPGenOrientMotorUWSolver
        makeDiagMats: FPUWMotorSolver
        stepUW: FPUWSolver
        Step: FPGenMotionMotorUWSolver
        RodStep: OpticalTrapMotionSolver
        Write: self
        makeDataframe: self

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init FPOpticalTrapGenMotionMotorUWSolver ->", end=" ")
        FPGenMotionMotorUWSolver.__init__(self, pfile, pdict)
        self.OTParseParams()
        self.calcOTInteractions(self.R1_pos,
                                self.R2_pos,
                                self.R1_vec,
                                self.R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: void, Create dataframe for output with optical trap data

        """
        FPGenMotionMotorUWSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        i_step = FPGenMotionMotorUWSolver.Write(self)
        self.OTWrite(i_step)
        return i_step
