#!/usr/bin/env python
from .FP_pass_ang_solver import FPPassiveAngSolver
from .FP_pass_CN_solver import FPPassiveCNSolver


"""@package docstring
File: FP_pass_ang_CN.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Class that solves the distribution of crosslinks in an angular system using the Crank-Nicolson method.
"""


class FPPassiveAngCNSolver(FPPassiveCNSolver, FPPassiveAngSolver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPPassiveAngCNSolver ->", end=" ")
        FPPassiveCNSolver.__init__(self, pfile, pdict)


##########################################
if __name__ == "__main__":
    pdes = FPPassiveAngCNSolver(sys.argv[1])
    pdes.Run()
    pdes.Save()
    # pdes.Save('FP_pass_LF.pickle')
