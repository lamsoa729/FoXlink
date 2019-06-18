#!/usr/bin/env python
from .solver import Solver
from .FP_initial_conditions import *


"""@package docstring
File: FP_static_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPStaticSolver(Solver):

    """!Docstring for FPRodMotionSolver. """

    def __init__(self, pfile=None, name="FP_static"):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param name: TODO

        """
        Solver.__init__(self, pfile=pfile, name=name)

    def Step(self):
        """!Step static solver forward in time using Strang splitting
            Add half value source matrix to xlink distribution
            Reduce resulting grid based on sink term
            Add half value of source matrix
        All of this can be done in one operation.
        @return: void

        """
        ko = self._params["ko"]
        dt = self.dt
        self.sgrid = ((dt / ko - (.5 * dt**2)) *
                      self.src_mat + (1. - ko * dt) * self.sgrid)


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
