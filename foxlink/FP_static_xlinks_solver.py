#!/usr/bin/env python
from .solver import Solver
from .FP_initial_conditions import *


"""@package docstring
File: FP_static_xlinks_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPStaticXlinksSolver(Solver):

    """! Class to solve the evolution of static xlinks bind and unbinding from solution."""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param name: TODO

        """
        print("Init FPStaticSolver ->", end=" ")
        Solver.__init__(self, pfile=pfile, pdict=pdict)

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
        # Strang splitting
        self.sgrid = ((dt - (.5 * ko * dt**2)) *
                      self.src_mat + (1. - ko * dt) * self.sgrid)
        # No strang splitting
        # self.sgrid = dt * self.src_mat + (1. - ko * dt) * self.sgrid


##########################################
if __name__ == "__main__":
    print("Not implemented yet")