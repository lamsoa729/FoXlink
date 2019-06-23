#!/usr/bin/env python
from .FP_gen_motion_solver import FPGenMotionSolver
from .FP_pass_CN_solver import FPPassiveCNSolver

"""@package docstring
File: FP_gen_mot_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionPassCNSolver (FPGenMotionSolver, FPPassiveCNSolver):

    """A PDE solver that incorporates diffusion and rod motion solved by Crank-Nicolson method for crosslink diffusion and forward Euler method for rod motion."""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system
        Note: parameter file needs viscosity, and xlink diffusion in order to run

        @param pfile: TODO
        @param name: TODO

        """
        print("Init FPGenMotionPassCNSolver ->", end=" ")
        FPGenMotionSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def Step(self):
        """Step forward in time with FPPassiveCN.Step and then use FPGenMotionSolver to step the change the position of rods.
        @return: TODO

        """
        # Update xlink positions
        FPPassiveCNSolver.Step(self)
        # Calculate new forces and torque
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force, self.torque, self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
