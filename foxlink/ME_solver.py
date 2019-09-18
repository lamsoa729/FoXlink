#!/usr/bin/env python
from solver.py import Solver
from scipy.integrate import solve_ivp, dblquad
import numpy as np
from numba import jit

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker-Planck equation
        in a series of moments of crosslinker head positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init MomentExpansionSolver -> ")
        Solver.__init__(self, pfile, pdict)

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: TODO

        """
        pass

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary
        @return: TODO

        """
        pass

    def makeDataframe(self):
        """!Create data frame to be written out
        @return: TODO

        """

    def Run(self):
        """!Run algorithm to solve system of ODEs
        @return: TODO

        """
        pass

    def Write(self):
        """!Write out data
        @return: TODO

        """
        pass
