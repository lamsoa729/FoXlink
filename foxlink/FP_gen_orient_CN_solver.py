#!/usr/bin/env python
from .FP_gen_orient_solver import FPGenOrientSolver
from .FP_pass_CN_solver import FPPassiveCNSolver


"""@package docstring
File: FP_gen_orient_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenOrientCNSolver(FPGenOrientSolver, FPPassiveCNSolver):

    """A PDE solver that incorporates crosslink motion through Crank-Nicolson integration method"""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init FPGenOrientCNSolver ->", end=" ")
        FPGenOrientSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()
