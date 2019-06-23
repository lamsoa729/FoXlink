#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
import sys
import os
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
# sys.path.append(os.path.join(os.path.dirname(__file__), '[PATH]'))


"""@package docstring
File: FP_gen_mot_CN_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionPassCNSolver (FPGenMotionSolver, FPPassiveCNSolver):

    """A PDE solver that incorporates diffusion and rod motion solved by Crank-Nicolson method for crosslink diffusion and forward Euler method for rod motion."""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system
        Note: parameter file needs viscosity, and xlink diffusion in order to run

        @param pfile: TODO
        @param name: TODO

        """
        print("Init FPGenMotionSolver ->", end=" ")
        FPGenMotionSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
