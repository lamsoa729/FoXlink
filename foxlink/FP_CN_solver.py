#!/usr/bin/env python
"""@package docstring
File: FP_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Abstract xlink algorithm class implementing Crank-Nicolson solving algorithm.
"""
from .FP_solver import FokkerPlanckSolver


class FPCNSolver(FokkerPlanckSolver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param pdict: parameter dictionary if no parameter file path

        """
        print("Init FPCNSolver ->", end=" ")
        FokkerPlanckSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeDiagMats(self):
        """!Make diagnal matrices for implicit and explicit steps of CN
        @return: void, modifies diagnol matricies

        """
        raise NotImplementedError(
            "makeDiagMats have not been defined for {}. To use the CN subclass, "
            "construction of these matrices is necessary.".format(
                self.__class__.__name__))
