#!/usr/bin/env python
from .gen_def_motion_solver import GenDefMotionSolver
from .pde_gen_motion_motor_uw_solver import PDEGenMotionMotorUWSolver

"""@package docstring
File: pde_gen_def_motion_motor_uw_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEGenDefMotionMotorUWSolver(
        GenDefMotionSolver, PDEGenMotionMotorUWSolver):

    """!Docstring for PDEGenDefMotionMotorUWSolver. """

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
        RodStep: GenDefMotionSolver
        Write: PDEGenOrientSolver
        makeDataframe: PDEGenOrientSolver

        @param pfile: parameter file path
        @param pdict: dictionary of parameters

        """
        print("Init PDEGenDefMotionMotorUWSolver ->", end=" ")
        PDEGenMotionMotorUWSolver.__init__(self, pfile, pdict)
        self.DefMotionParseParams()

    def RodStep(self, *args, **kwargs):
        return GenDefMotionSolver.RodStep(self, *args, **kwargs)
