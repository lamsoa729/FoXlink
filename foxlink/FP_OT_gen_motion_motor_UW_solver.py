#!/usr/bin/env python
from .optical_trap_gen_mot_solver import OpticalTrapGenMotionSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver


"""@package docstring
File: FP_OT_gen_motion_motor_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPOpticalTrapGenMotionMotorUWSolver(
        OpticalTrapGenMotionSolver, FPGenMotionMotorUWSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: OpticalTrapGenMotionSolver (specified)
        calcTorqueMat: OpticalTrapGenMotionSolver (specified)
        calcVelocityMats: FPGenOrientMotorUWSolver
        makeDiagMats: FPUWMotorSolver
        stepUW: FPUWSolver
        Step: FPUWMotorSolver
        RodStep: OpticalTrapGenMotionSolver (specified)
        Write: OpticalTrapGenMotionSolver (specified)
        makeDataframe: self

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPOpticalTrapGenMotionMotorUWSolver ->", end=" ")
        FPGenMotionMotorUWSolver.__init__(self, pfile, pdict)
        self.OTParseParams()

    def calcForceMatrix(self):
        OpticalTrapGenMotionSolver.calcForceMatrix(self)

    def calcTorqueMatrix(self):
        OpticalTrapGenMotionSolver.calcTorqueMatrix(self)

    def RodStep(self, force1=0, force2=0, torque1=0, torque2=0,
                R1_pos=None, R2_pos=None, R1_vec=None, R2_vec=None):
        OpticalTrapGenMotionSolver.RodStep(self, force1, force2, torque1, torque2,
                                           R1_pos, R2_pos, R1_vec, R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: TODO

        """
        FPGenMotionMotorUWSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        OpticalTrapGenMotionSolver.Write()
