#!/usr/bin/env python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from copy import deepcopy as dcp
# from FP_helpers import *
from .FP_CN_solver import FPCNSolver


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPPassiveCNSolver(FPCNSolver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param name: name to store data under

        """
        print("Init FPPassiveCNSolver ->", end=" ")
        FPCNSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeDiagMats(self):
        """!Make matrices to necessary to carry out explicit and implicit
        time step in simulation
        @return: TODO

        """
        # Parameters for matrices
        g_xl = self._params["gamma_xl"]
        D = 1. / (self._params["beta"] * g_xl)
        ko = self._params["ko"]
        dt = self.dt
        # Create main matrix diagnol
        diagR = np.array(
            [1. - (dt * ((.5 * ko) + (D / (self.ds**2))))] * max(self.ns1, self.ns2))
        diagL = np.array(
            [1. + (dt * ((.5 * ko) + (D / (self.ds**2))))] * max(self.ns1, self.ns2))
        # Create diagnols displaced 1 index away from the main matrix diagnol
        off_setR = np.array([.5 * dt * D / (self.ds**2)]
                            * max(self.ns1, self.ns2))
        off_setRtop = dcp(off_setR)
        off_setRbottom = dcp(off_setR)
        # Neumann boundary conditions
        #       (not at i=0 and i=-1 because these get cut off by dia_matrix)
        off_setRtop[1] *= 2
        off_setRbottom[-2] *= 2

        off_setL = -1. * off_setR
        off_setLtop = dcp(off_setL)
        off_setLbottom = dcp(off_setL)
        # Neumann boundary conditions
        #       (not at i=0 and i=-1 because these get cut off by dia_matrix)
        off_setLtop[1] *= 2
        off_setLbottom[-2] *= 2
        # Initialize diagnol arrays using scipy dia_matrix. Look up on web
        diagL_arr = np.stack((diagL, off_setLtop, off_setLbottom))
        diagR_arr = np.stack((diagR, off_setRtop, off_setRbottom))
        offsets = [0, 1, -1]
        # Actual initialization
        self.DsqrRight = sparse.dia_matrix((diagR_arr, offsets),
                                           shape=(self.ns1, self.ns2)).tocsc()
        self.DsqrRightT = self.DsqrRight.T
        DsqrLeft = sparse.dia_matrix((diagL_arr, offsets),
                                     shape=(self.ns1, self.ns2)).tocsc()
        self.DsqrLeftInv = inv(DsqrLeft)
        self.DsqrLeftInvT = self.DsqrLeftInv.T.tocsc()

    def Step(self):
        """!Step forward once in time using alternating implicit method
        @return: TODO

        """

        # Step 1: Add half the source term to current solution
        sgrid_bar = .5 * self.dt * self.src_mat + self.sgrid
        # Steps 2 & 3: Apply crank-nicolson solver
        sgrid_bar = self.stepCN(sgrid_bar)
        # Step 4: Add the other half of the source term
        self.sgrid = .5 * self.dt * self.src_mat + sgrid_bar

    def stepCN(self, sgrid_bar):
        """!Alternating implicit Crank-Nicolson algorithmkstep of code

        @param sgrid_bar: TODO
        @return: TODO

        """
        # Step 2a: Explicit step along s2 direction
        sgrid_bar = sparse.csc_matrix.dot(sgrid_bar, self.DsqrRightT)
        # Step 2b: Implicit step along s1 direction
        sgrid_bar = sparse.csc_matrix.dot(self.DsqrLeftInv, sgrid_bar)

        # Step 3a: Explicit step along s1 direction
        sgrid_bar = sparse.csc_matrix.dot(self.DsqrRight, sgrid_bar)
        # Step 3b: Implicit step along s2 direction
        sgrid_bar = sparse.csc_matrix.dot(sgrid_bar, self.DsqrLeftInvT)
        return sgrid_bar


##########################################
if __name__ == "__main__":
    pdes = FPPassiveCNSolver(sys.argv[1])
    pdes.Run()
    pdes.Save()
    # pdes.Save('FP_pass_LF.pickle')
