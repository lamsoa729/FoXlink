#!/usr/bin/env python
from .FP_gen_orient_solver import FPGenOrientSolver, reparameterize_rods
from .FP_rod_motion_solver import FPRodMotionSolver


"""@package docstring
File: FP_gen_motion_Solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionSolver(FPGenOrientSolver, FPRodMotionSolver):

    """Docstring for FPGenMotionSolver. """

    def __init__(self, pfile=None, name="FP_gen_motion"):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param name: TODO

        """


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
