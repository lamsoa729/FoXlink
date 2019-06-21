#!/usr/bin/env python
from .FP_gen_orient_solver import FPGenOrientSolver
from .FP_rod_motion_solver import FPRodMotionSolver


"""@package docstring
File: FP_gen_motion_Solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Combination class of general orientation solver and rod motion solver
"""


class FPGenMotionSolver(FPGenOrientSolver, FPRodMotionSolver):

    """Docstring for FPGenMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init FPGenMotionSolver ->", end=" ")
        FPGenOrientSolver.__init__(self, pfile=pfile, pdict=pdict)


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
