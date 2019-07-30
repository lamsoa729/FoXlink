#!/usr/bin/env python

"""@package docstring
File: FP_OT_gen_motion_static_xlinks_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from .optical_trap_motion_solver import OpticalTrapMotionSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionStaticXlinksSolver


class FPOpticalTrapGenMotionStaticXlinksSolver(
        OpticalTrapMotionSolver, FPGenMotionStaticXlinksSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: FPGenOrientSolver
        calcTorqueMat: FPGenOrientSolver
        Step: FPGenMotionStaticXlinksSolver
        RodStep: OpticalTrapMotionSolver
        Write: self
        makeDataframe: self

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init FPOpticalTrapGenMotionStaticXlinksSolver ->", end=" ")
        FPGenMotionStaticXlinksSolver.__init__(self, pfile, pdict)
        self.OTParseParams()
        self.calcOTInteractions(self.R1_pos,
                                self.R2_pos,
                                self.R1_vec,
                                self.R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: void, Create dataframe for output with optical trap data

        """
        FPGenMotionStaticXlinksSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        i_step = FPGenMotionStaticXlinksSolver.Write(self)
        self.OTWrite(i_step)
        return i_step
