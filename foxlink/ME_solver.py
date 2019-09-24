#!/usr/bin/env python
from solver.py import Solver
from scipy.integrate import solve_ivp, dblquad
from .ME_helpers import evolver_zrl
from rod_motion_solver import get_rod_drag_coeff
import numpy as np

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def choose_ODE_solver(sol, t, vo, fs, ko, c, ks, beta, L1, L2, d, visc):
    """!Create a closure for ode solver

    @param sol: Array of time-dependent variables in the ODE
    @param t: time
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param d: Diameter of rods
    @param visc: Viscocity of surrounding fluid
    @return: TODO

    """
    # Get drag coefficients
    gpara1, gperp1, grot1 = get_rod_drag_coeff(visc, L1, d)
    gpara2, gperp2, grot2 = get_rod_drag_coeff(visc, L2, d)

    def evolver_zrl_closure(sol, t):
        """!Define the function of an ODE solver with certain constant
        parameters.

        @param sol: TODO
        @param t: TODO
        @return: TODO

        """
        r1 = sol[:3]
        r2 = sol[3:6]
        u1 = sol[6:9]
        u2 = sol[9:12]
        return evolver_zrl(r1, r2, u1, u2,  # Vectors
                           sol[12], sol[13], sol[14],  # Moments
                           sol[15], sol[16], sol[17],
                           gpara1, gperp1, grot1,  # Friction coefficients
                           gpara2, gperp2, grot2,
                           vo, fs, ko, c, ks, beta, L1, L2)  # Other parameters

    return evolver_zrl_closure


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker - Planck equation
        in a series of moments of motor end positions on rods.
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
