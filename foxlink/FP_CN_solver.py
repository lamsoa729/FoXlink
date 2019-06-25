#!/usr/bin/env python
from .solver import Solver

"""@package docstring
File: FP_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPCNSolver(Solver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPCNSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeDiagMats(self):
        """!Make diagnal matrices for implicit solving
        @return: TODO

        """
        raise NotImplementedError(
            "makeDiagMats have not been defined for {}. To use the CN subclass,  construction of these matrices is necessary.".format(
                self.__class__.__name__))


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
