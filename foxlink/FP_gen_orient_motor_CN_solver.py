#!/usr/bin/env python
from .FP_gen_orient_CN_solver import FPGenOrientCNSolver
from .FP_CN_solver import FPCNSolver


"""@package docstring
File: FP_gen_orient_motor_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenOrientMotorCNSolver(FPGenOrientCNSolver, FPCNSolver):

    """A PDE solver that incorporates crosslink motion through Crank-Nicolson integration method"""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init FPGenOrientCNSolver ->", end=" ")
        FPGenOrientCNSolver.__init__(self, pfile, pdict)

    def makeDiagMats(self):
        """!TODO: Docstring for makeDiagMats.
        @return: TODO

        """
        # Create stretch matrix
        # Create velocity matrix
        # Create derivative velocity matrix
        # Create numerical derivative matrix
        pass

    def Step(self):
        """!Step using methods
        @return: TODO

        """
        # Add half of the source matrix
        # First implicit step
        # Second implicit step
        # Add half of the source matrix
        pass


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
