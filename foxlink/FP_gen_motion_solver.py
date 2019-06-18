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
from FP_gen_orient_solver import FPGenOrientSolver, reparameterize_rods
from FP_rod_motion_solver import FPRodMotionSolver


"""@package docstring
File: FP_gen_motion_Solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionSolver(FPGenOrientSolver, FPRodMotionSolver):

    """Docstring for FPGenMotionSolver. """

    def __init__(self, pfile=None, name="FP_gen_motion"):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param name: TODO

        """


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
