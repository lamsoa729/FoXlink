#!/usr/bin/env python
from .pde_gen_motion_solver import PDEGenMotionSolver
from .pde_pass_cn_solver import PDEPassiveCNSolver

"""@package docstring
File: pde_gen_motion_pass_cn_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEGenMotionPassCNSolver(PDEGenMotionSolver, PDEPassiveCNSolver):

    """A PDE solver that incorporates diffusion and rod motion solved by
    Crank-Nicolson method for crosslink diffusion and forward Euler
    method for rod motion."""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system
        Note: parameter file needs viscosity and xlink diffusion in order to run

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init PDEGenMotionPassCNSolver ->", end=" ")
        PDEGenMotionSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def Step(self):
        """Step forward in time with PDEPassiveCN.Step and then use PDEGenMotionSolver to step the change the position of rods.
        @return: TODO

        """
        # Update xlink positions
        PDEPassiveCNSolver.Step(self)
        # Calculate new forces and torque
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions and recalculate source matrices
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force1, self.force2, self.torque1, self.torque2,
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
