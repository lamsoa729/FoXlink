#!/usr/bin/env python
from .optical_trap_gen_mot_solver import OpticalTrapGenOrientSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver


"""@package docstring
File: FP_OT_gen_motion_motor_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPOpticalTrapGenMotionMotorUWSolver(
        OpticalTrapGenOrientSolver, FPGenMotionMotorUWSolver):
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
        RodStep: self
        Write: OpticalTrapGenMotionSolver (specified)
        makeDataframe: self

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPOpticalTrapGenMotionMotorUWSolver ->", end=" ")
        self._pfile = pfile
        self._params = pdict
        self.ParseParams(skip=True)
        self.OTParseParams()
        FPGenMotionMotorUWSolver.__init__(self, pfile, pdict)

    def calcForceMatrix(self):
        OpticalTrapGenOrientSolver.calcForceMatrix(self)

    def calcTorqueMatrix(self):
        OpticalTrapGenOrientSolver.calcTorqueMatrix(self)

    def RodStep(self, force1=0, force2=0, torque1=0, torque2=0,
                R1_pos=None, R2_pos=None, R1_vec=None, R2_vec=None):
        """! Change the position of rods based on forces and torques exerted on rod
        @param force: Force vector of rod2 by rod1
        @param torque: Torque vector of rod2 by rod1
        @param R1_pos: TODO
        @param R2_pos: TODO
        @param R1_vec: TODO
        @param R2_vec: TODO
        @return: void
        @return: TODO

        """
        return FPGenMotionMotorUWSolver.RodStep(self,
                                                force1 + self.ot1_force,
                                                force2 + self.ot2_force,
                                                torque1 + self.ot1_torque,
                                                torque2 + self.ot2_torque,
                                                R1_pos, R2_pos, R1_vec, R2_vec)

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: TODO

        """
        FPGenMotionMotorUWSolver.makeDataframe(self)
        self.addOTDataframe()

    def Write(self):
        OpticalTrapGenOrientSolver.Write(self)
