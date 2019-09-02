#!/usr/bin/env python
from .gen_def_motion_solver import GenDefMotionSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver

"""@package docstring
File: FP_gen_def_motion_motor_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenDefMotionMotorUWSolver(
        GenDefMotionSolver, FPGenMotionMotorUWSolver):

    """!Docstring for FPGenDefMotionMotorUWSolver. """

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
        RodStep: GenDefMotionSolver
        Write: FPGenOrientSolver
        makeDataframe: FPGenOrientSolver

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init FPGenDefMotionMotorUWSolver ->", end=" ")
        FPGenMotionMotorUWSolver.__init__(self, pfile, pdict)
        self.DefMotionParseParams()

    def RodStep(self, *args, **kwargs):
        return GenDefMotionSolver.RodStep(self, *args, **kwargs)
