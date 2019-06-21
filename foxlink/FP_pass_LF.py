#!/usr/bin/env python
import sys
from .FP_helpers import laplace_5p, boltz_fact_ang
from .solver import Solver
from .FP_pass_ang_solver import FPPassiveAngSolver


"""@package docstring
File: FP_pass_LF.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

#  TODO: Speed up by using matrix math <21-06-19, ARL> #


class FPPassiveLFSolver(FPPassiveAngSolver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the leap frog method with 4 point laplacian.
    """

    def __init__(self, pfile=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path

        """
        print("Init FPPassiveLFSolver ->", end=" ")
        FPPassiveAngSolver.__init__(self, pfile)

    def Step(self):
        """!Step solver using midpoint (leap frog) method.
        @return: void

        """
        # Set up variables for solving function
        g_xl = self._params["gamma"]  # Friction coefficient of xlinks
        beta = self._params["beta"]  # Inverse temp * k_b
        D = 1. / (self._params["beta"] * g_xl)
        ds = self.ds
        ko = self._params["ko"]
        ks = self._params['ks']
        ho = self._params['ho']
        co = self._params['co']
        # Create solution grid to swap out at end of sim
        sgrid_n = np.zeros((self.ns1, self.ns2))
        # self.calcForceGrid(self.phio)  # Stationary for now
        for i in range(1, self.ns1 - 1):
            for j in range(1, self.ns2 - 1):
                # Diffusion term
                d_sol_ij = D * laplace_5p(i, j, self.sgrid_arr[1], ds)
                # # Drag force on head 1 (leap frog derivative)
                # d_sol_ij -= (1. / (2. * ds * g_xl)) * (
                #     (self.fgrid_arr[0][i + 1, j] * self.sgrid_arr[1][i + 1, j]) -
                #     (self.fgrid_arr[0][i - 1, j] * self.sgrid_arr[1][i - 1, j]))
                # # Drag force on head 2 (leap frog derivative)
                # d_sol_ij -= (1. / (2. * ds * g_xl)) * (
                #     (self.fgrid_arr[1][i, j + 1] * self.sgrid_arr[1][i, j + 1]) -
                #     (self.fgrid_arr[1][i, j - 1] * self.sgrid_arr[1][i, j - 1]))
                # Source and sink terms
                d_sol_ij += ko * (co * boltz_fact_ang(self.s1[i],
                                                      self.s2[j],
                                                      self.phio,
                                                      ks, ho, beta) -
                                  self.sgrid_arr[1][i, j])
                # Evolve solution at ij
                sgrid_n[i, j] = self.sgrid_arr[0][i, j] + (
                    2. * self.dt * d_sol_ij)

        # Shift grids, not sure if the reference will be passed correctly
        self.sgrid_arr[0] = self.sgrid_arr[1]
        self.sgrid_arr[1] = sgrid_n


##########################################
if __name__ == "__main__":
    pdes = FPPassiveLFSolver(sys.argv[1])
    pdes.Run()
