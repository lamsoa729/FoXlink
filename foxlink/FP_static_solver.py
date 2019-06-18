#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
# Testing
# import pdb
# import time, timeit
# import line_profiler
# Analysis
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import h5py
# import yaml
# from math import *
# Speed
# from numba import jit
# Other importing
from solver import Solver
import FP_initial_conditions as IC


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
