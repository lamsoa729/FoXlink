#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
# Testing
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

from FP_gen_motion_solver import FPGenMotionSolver
from FP_gen_orient_solver import FPGenOrientSolver
from FP_static_solver import FPStaticSolver


"""@package docstring
File: FP_gen_motion_static_xlinks.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenMotionStaticXlinks(FPGenMotionSolver, FPStaticSolver):

    """Solver class to calculate rod motion in a solution of crosslinks binding and unbinding."""

    def __init__(self, pfile=None, name="FP_rod_motion"):
        """TODO: to be defined1. """
        print("Init FPGenMotionStaticXlinks ->", end=" ")
        FPGenOrientSolver.__init__(self, pfile=pfile,
                                   name="FP_gen_motion_static_xlinks")

    def Step(self):
        """Step forward in time with FPStaticStep and then use FPGenMotionSolver to step the change the position of rods.
        @return: TODO

        """
        # Update xlink positions
        FPStaticSolver.Step(self)
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        # Update rod positions
        self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec = self.RodStep(
            self.force, self.torque, self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
