#!/usr/bin/env python
from .optical_trap_motion_solver import OpticalTrapMotionSolver
from .FP_gen_motion_motor_UW_solver import FPGenOrientMotorUWSolver


"""@package docstring
File: FP_OT_gen_motion_motor_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPOpticalTrapGenMotionMotorUWSolver(
        OpticalTrapMotionSolver, FPGenOrientMotorUWSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        ParseParams: FPGenOrientSolver
        calcSourceMat: FPGenOrientSolver
        calcForceMat: FPGenOrientSolver
        calcTorqueMat: FPGenOrientSolver
        calcVelocityMats: FPGenOrientMotorUWSolver
        makeDiagMats: FPUWMotorSolver
        stepUW: FPUWSolver
        Step: FPUWMotorSolver
        RodStep: OpticalTrapMotionSolver
        Write: self
        makeDataframe: self

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPOpticalTrapGenMotionMotorUWSolver ->", end=" ")
        self._pfile = pfile
        self._params = pdict
        FPGenOrientMotorUWSolver.__init__(self, pfile, pdict)
        self.OTParseParams()
        self.calcOTInteractions(self.R1_pos,
                                self.R2_pos,
                                self.R1_vec,
                                self.R2_vec)

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
            self.force1, self.force1, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
        # Update
        self.calcVelocityMats()

    def makeDataframe(self):
        """! Make data frame with optical trap objects
        @return: TODO

        """
        FPGenOrientMotorUWSolver.makeDataframe(self)
        self.addOTDataframe()
