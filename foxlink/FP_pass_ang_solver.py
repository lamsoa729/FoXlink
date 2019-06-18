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


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPPassiveAngSolver(Solver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the leap frog method with 4 point laplacian.
    """

    def __init__(self, pfile=None, name='FP_pass_ang'):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path

        """
        Solver.__init__(self, pfile, name)

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.
        @return: void
        """
        Solver.ParseParams(self)
        self.phio = self._params["phio"]  # Angle between MTs

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """
        Solver.makeDataframe(self)
        self._phi_dset = self._mt_grp.create_dataset(
            'phi', shape=(self._nframes, 1))

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        i_step = Solver.Write(self)
        self._phi_dset[i_step] = self.phio
        # self.df["xl_dens"] += [self.sgrid.tolist()]


##########################################
if __name__ == "__main__":
    pdes = FPPassiveLFSolver(sys.argv[1])
    pdes.Run()
    pdes.Save('FP_pass_LF.pickle')
