#!/usr/bin/env python


"""@package docstring
File: FP_UW_motor_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from .FP_UW_solver import FPUWSolver
from scipy import sparse


"""@package docstring
File: FP_UW_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPUWMotorSolver(FPUWSolver):

    """!Solve the Fokker-Planck equation using force-dependent velocity relation
    and the upwind integration method.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param pdict: parameter dictionary

        """
        print("Init FPUWMotorSolver ->", end=" ")
        FPUWSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeDiagMats(self):
        """!Make diagnal matrices for implicit solving
        @return: TODO

        """
        # Main diagnol
        diag = np.ones(max(self.ns1, self.ns2))
        # Offset diagnol from previous position
        off_set_diag = -1. * np.ones(max(self.ns1, self.ns2))
        # Neumann boundary conditions
        #   No flux from the start of rod. Nothing needs to be done
        #   No flux leaving from end of rod. Set last term of diagnol to zero.
        diag[-1] = 0

        # Create matrix using sparse matrices
        diag_arr = np.stack(diag, off_set_diag)
        off_sets = [0, -1]
        self.diagGradUW = sparse.diag_matrix((diag_arr, off_set_diag),
                                             shape=(self.ns1, self.ns2)).tocsc()
        self.diagGradUWT = self.diagGradUW.T

    def makeVelocityMats(self):
        """!TODO: Docstring for makeVelocityMat.
        @return: TODO

        """
        raise NotImplementedError(("makeVelocityMat has not been defined for {}.",
                                   " To use the UWMotor subclass, construction,",
                                   "of these matrices is necessary.").format(self.__class__.__name__))

    def Step(self):
        """!Step forward once in time using Upwind method and source terms
        @return: TODO

        """

        ko = self._params['ko']
        # Add half the source terms to current solution
        sgrid_bar = .5 * self.dt * self.src_mat + self.sgrid
        # Take away half the sink term
        sgrid_bar *= (1. - ko * self.dt * .5) *
        # Apply upwind solver
        sgrid_bar = self.stepUW(sgrid_bar)
        # Take away the other half of the sink term
        sgrid_bar *= (1. - ko * self.dt * .5)
        # Step 4: Add the other half of the source
        self.sgrid = .5 * self.dt * self.src_mat + sgrid_bar
