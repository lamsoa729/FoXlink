#!/usr/bin/env python

"""@package docstring
File: pde_ot_gen_motion_static_xlinks_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from .optical_trap_motion_solver import OpticalTrapMotionSolver
from .pde_gen_motion_static_xlinks_solver import PDEGenMotionStaticXlinksSolver


class PDEOpticalTrapGenMotionStaticXlinksSolver(
        OpticalTrapMotionSolver, PDEGenMotionStaticXlinksSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: PDEGenOrientSolver
        calcSourceMat: PDEGenOrientSolver
        calcForceMat: PDEGenOrientSolver
        calcTorqueMat: PDEGenOrientSolver
        Step: PDEGenMotionStaticXlinksSolver
        RodStep: OpticalTrapMotionSolver
        Write: self
        makeDataframe: self

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init PDEOpticalTrapGenMotionStaticXlinksSolver ->", end=" ")
        PDEGenMotionStaticXlinksSolver.__init__(self, pfile, pdict)
        self.OTParseParams()
        self.calcOTInteractions(self.R1_pos,
                                self.R2_pos,
                                self.R1_vec,
                                self.R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: void, Create dataframe for output with optical trap data

        """
        PDEGenMotionStaticXlinksSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        i_step = PDEGenMotionStaticXlinksSolver.Write(self)
        self.OTWrite(i_step)
        return i_step
