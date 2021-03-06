#!/usr/bin/env python

from .pde_uw_solver import PDEUWSolver

"""@package docstring
File: pde_motor_uw_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEMotorUWSolver(PDEUWSolver):

    """!Solve the Fokker-Planck equation using force-dependent velocity relation
    and the upwind integration method.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param pdict: parameter dictionary

        """
        print("Init PDEMotorUWSolver ->", end=" ")
        PDEUWSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()

    def makeVelocityMats(self):
        """!TODO: Docstring for makeVelocityMat.
        @return: TODO

        """
        raise NotImplementedError(("makeVelocityMat has not been defined for {}."
                                   " To use the UWMotor subclass, construction"
                                   "of these matrices is necessary.").format(self.__class__.__name__))

    def Step(self):
        """!Step forward once in time using Upwind method and source terms
        @return: TODO

        """

        ko = self._params['ko']
        self.checkCFL()
        # Add half the source terms to current solution
        sgrid_bar = .5 * self.dt * self.src_mat + self.sgrid
        # Take away half the sink term
        sgrid_bar *= (1. - ko * self.dt * .5)
        # Apply boundary conditions before using differential matrix
        if self._params["boundary_conditions"] == 'zero':
            self.apply_dirichlet_bc()
        # Apply upwind solver
        sgrid_bar = self.stepUW(sgrid_bar, self.vel_mat1, self.vel_mat2)
        # Take away the other half of the sink term
        sgrid_bar *= (1. - ko * self.dt * .5)
        # Add the other half of the source
        self.sgrid = .5 * self.dt * self.src_mat + sgrid_bar
        # Apply boundary conditions once again
        if self._params["boundary_conditions"] == 'zero':
            self.apply_dirichlet_bc()

    def checkCFL(self):
        """!Check to make sure CFL conditions is satisfied
        @return: void

        """
        if (self._params['vo'] * self.dt) >= self.ds:
            raise RuntimeError("CFL condition (v < ds/dt) is not satisfied.",
                               "The parameter dt must be reduced or ds increased.")
