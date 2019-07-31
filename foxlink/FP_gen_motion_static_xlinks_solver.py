#!/usr/bin/env python
from .FP_gen_motion_solver import FPGenMotionSolver
from .FP_static_xlinks_solver import FPStaticXlinksSolver


"""@package docstring
File: FP_gen_motion_static_xlinks.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Solver class to calculate rod motion in a solution of crosslinks binding and unbinding.
"""


class FPGenMotionStaticXlinksSolver(FPGenMotionSolver, FPStaticXlinksSolver):

    """Solver class to calculate rod motion in a solution of crosslinks binding and unbinding."""

    def __init__(self, pfile=None, pdict=None):
        """Initialize FPGenMotionStaticXlinks solver using FPGenMotionSolver"""
        print("Init FPGenMotionStaticXlinks ->", end=" ")
        FPGenMotionSolver.__init__(self, pfile=pfile, pdict=pdict)

    def Step(self):
        """Step forward in time with FPStaticStep and then use FPGenMotionSolver to step the change the position of rods.
        @return: void, steps the sgrid, R1_pos, R1_vec, R2_pos, R2_vec. Calculates forces and torques.

        """
        # Update xlink positions
        FPStaticXlinksSolver.Step(self)
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force1, self.force2, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
