#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
import sys
import os
# Testing
import pdb
import time
import timeit
# import line_profiler
# Analysis
from math import *
from copy import deepcopy as dcp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle as pickle
from scipy.integrate import dblquad, odeint
import yaml
# from math import *
# Speed
from numba import jit
from FP_helpers import *
from solver import Solver
# Other importing
# sys.path.append(os.path.join(os.path.dirname(__file__), '[PATH]'))


"""@package docstring
File: FP_passive_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Solve Fokker-Planck equation
"""

default_params = {
    "phio": 1.570796327,  # Initial starting angle of phi
    "L1": 100,  # Length of microtubule 1
    "L2": 100,  # Length of microtubule 2
    # "dt": 1,  # Time step
    "nsteps": 1000000,  # Total number of steps, used to define dt if dt is ND
    "nt": 2000,  # total time
    "ds": 1,  # Segmentation size of microtubules
    "ko": 1,  # Crosslinker turnover rate
    "co": 1,  # Effective crosslinker concentration
    "ks": 1,  # Crosslinker spring concentration
    "ho": 1,  # Equilibrium length of crosslinkers
    # "vo": 1,  # Base velocity of crosslinker heads
    # "fs": 1,  # Stall force of crosslinker heads
    "beta": 1,  # Inverse temperature
}


@jit
def dxl_dens_dt(y, t, s1, s2, ko, co, ks, ho, beta, omega=0):
    """!Calculate the partial derivative of xl_dens with respect to time based on
    previous xl_dens calculation

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param ko: TODO
    @param ks: TODO
    @param ho: TODO
    @param beta: TODO
    @return: TODO

    """
    xl_denso, phi = y
    dphi = omega
    if xl_denso < 0.0:
        xl_denso = 0.0
    b_fact = boltz_fact_ang(s1, s2, phi, ks, ho, beta)

    return [ko * (co * b_fact - xl_denso), dphi]


class PassiveSolver(Solver):

    """!Docstring for PassiveSolver. """

    def __init__(self, pfile=None):
        """!Set parameters for PDE to be solved including boundary conditions

        @param pfile: parameter file for PDEs

        """
        self.solved_flag = False
        Solver.__init__(self, pfile)

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.

        @return: TODO

        """
        Solver.ParseParams(self)
        self.phio = self._params["phio"]  # Angle between MTs

    def makeSolutionGrid(self):
        Solver.makeSolutionGrid(self)
        self.sgrid = self.sgrid.tolist()

    def Run(self):
        """!Run PDE solver with parameters in pfile
        @return: void

        """
        if self._params['type'] == 'aligning':
            self.omega = (float(-self._params['phio']) / float(self.nt)) * .8
        else:
            self.omega = 0.

        for i in range(0, self.ns1):
            for j in range(0, self.ns2):
                t0 = time.time()
                f_args = (self.s1[i], self.s2[j],
                          self._params["ko"], self._params["co"],
                          self._params["ks"], self._params["ho"],
                          self._params["beta"], self.omega)
                soln = odeint(
                    dxl_dens_dt,
                    (self.sgrid[i][j], self._params['phio']),
                    self.time,
                    args=f_args)
                self.sgrid[i][j] = soln[:, 0]
                self.phi = soln[:, 1]
                t1 = time.time()

        solved_flag = True
        self.makeDataframe()
        return

    def makeDataframe(self):
        """!Changes sgrid so that 1st index is step number, second index
        @return: TODO

        """
        if self.solved_flag:
            self.df = {"params": self._params}
            time = self.time[::self.nwrite]
            self.df["time"] = time
            self.df["s1"] = self.s1.tolist()
            self.df["s2"] = self.s2.tolist()
            self.df["xl_dens"] = []
            xl_dens_n = np.zeros((self.ns1, self.ns2)).tolist()
            for n, t in enumerate(time):
                for i in range(0, self.ns1):
                    for j in range(0, self.ns2):
                        xl_dens_n[i][j] = self.sgrid[i][j][n * self.nwrite]
                self.df["xl_dens"] += [dcp(xl_dens_n)]
            self.df['phi'] = self.phi[::self.nwrite]
        else:
            pass

    def Save(self, filename="FP_passive.pickle"):
        """!Create a pickle file of solution
        @return: void

        """
        with open(filename, 'wb') as f:
            pickle.dump(self.df, f, -1)


##########################################
if __name__ == "__main__":
    pdes = PassiveSolver(sys.argv[1])
    pdes.Run()
    pdes.makeDataframe()
    pdes.Save()
