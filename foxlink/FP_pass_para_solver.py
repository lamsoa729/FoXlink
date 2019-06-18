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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import yaml
import h5py
from scipy import sparse
from copy import deepcopy as dcp
# from math import *
# Speed
from numba import jit
# Other importing
from FP_helpers import *
from solver import Solver
from FP_initial_conditions import *


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPPassiveParaSolver(Solver):

    """!Solve the Fokker-Planck equation for passive crosslinkers with
    a parallel geometry.
    """

    def __init__(self, pfile=None, name='FP_passive_para'):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path

        """
        Solver.__init__(self, pfile, name)

    def ParseParams(self):
        """! Parse parameters from file and add to member variables
        @return: void
        """
        Solver.ParseParams(self)
        # Separation vector of MT centers
        self.R_pos = self._params['R_pos']

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """
        Solver.makeDataframe(self)
        self._R_dset = self._mt_grp.create_dataset(
            'R_pos', shape=(self._nframes, 3))

    def makeSourceMatrix(self):
        """TODO: Docstring for makeSourceMatrix.
        @return: TODO

        """
        self.src_mat = self._params['ko'] * make_para_source_mat(self.s1, self.s2,
                                                                 self.R_pos,
                                                                 self._params['co'],
                                                                 self._params['ks'],
                                                                 self._params['ho'],
                                                                 self._params["beta"])

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        i_step = Solver.Write(self)
        self._R_dset[i_step] = self.R_pos


##########################################
if __name__ == "__main__":
    pass
