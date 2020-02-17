#!/usr/bin/env python
"""@package docstring
File: pde_gen_motion_static_xlinks.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Solver class to calculate rod motion in a solution of crosslinks binding and unbinding.
"""
from .pde_gen_motion_solver import PDEGenMotionSolver
from .pde_static_xlinks_solver import PDEStaticXlinksSolver


class PDEGenMotionStaticXlinksSolver(
        PDEGenMotionSolver, PDEStaticXlinksSolver):

    """Solver class to calculate rod motion in a solution of crosslinks binding and unbinding."""

    def __init__(self, pfile=None, pdict=None):
        """Initialize PDEGenMotionStaticXlinks solver using PDEGenMotionSolver"""
        print("Init PDEGenMotionStaticXlinks ->", end=" ")
        PDEGenMotionSolver.__init__(self, pfile=pfile, pdict=pdict)

    def Step(self):
        """Step forward in time with PDEStaticStep and then use PDEGenMotionSolver to step the change the position of rods.
        @return: void, steps the sgrid, R1_pos, R1_vec, R2_pos, R2_vec. Calculates forces and torques.

        """
        # Update xlink positions
        PDEStaticXlinksSolver.Step(self)
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force1, self.force2, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
