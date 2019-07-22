#!/usr/bin/env python
from .solver import Solver
from scipy import sparse
import numpy as np


"""@package docstring
File: FP_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPUWSolver(Solver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param pdict: parameter dictionary if file is not given directly.

        """
        print("Init FPUWSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeDiagMats(self):
        """!Make diagnal matrices for implicit solving
        @return: TODO

        """

        # Main diagnol of differential matrix
        diag = np.ones(max(self.ns1, self.ns2))
        # Offset diagnol in lower triangle of differential matrix
        off_set_diag = -1. * np.ones(max(self.ns1, self.ns2))
        if "end_pause" in self._params:
            if self._params["end_pause"]:
                print("End pausing")
                # Neumann boundary conditions
                #   No flux from the start of rod. Nothing needs to be done
                #   No flux leaving from end of rod. Set last term of main
                #   diagnol to zero.
                diag[-1] = 0
            else:
                print("No end pausing")
                # End flux term diffuses off the end of the rod

        # Create matrix using sparse numpy matrices
        diag_arr = np.stack((diag, off_set_diag))
        off_sets = [0, -1]
        self.diagGradUW = (1. / self.ds) * sparse.dia_matrix((diag_arr, off_sets),
                                                             shape=(self.ns1, self.ns2)).tocsc()
        self.diagGradUWT = self.diagGradUW.T

    def stepUW(self, sgrid_bar):
        """!Step crosslink density forward in time using upwind method.

        @param sgrid_bar: Current solution to differential equations before
                          upwind integration is applied
        @return: Current solution after upwind method

        """
        #  TODO: TEST using pytest <26-06-19, ARL> #
        # Explicit step along s1 and s2 direction with corresponding velocities
        return -1. * self.dt * (
            sparse.csc_matrix.dot(self.diagGradUW,
                                  np.multiply(self.vel_mat1, sgrid_bar)) +
            sparse.csc_matrix.dot(np.multiply(self.vel_mat2, sgrid_bar),
                                  self.diagGradUWT)) + sgrid_bar
