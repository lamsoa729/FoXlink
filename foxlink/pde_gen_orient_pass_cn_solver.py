#!/usr/bin/env python
from .pde_gen_orient_solver import PDEGenOrientSolver
from .pde_pass_cn_solver import PDEPassiveCNSolver


"""@package docstring
File: pde_gen_orient_pass_cn_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class PDEGenOrientPassCNSolver(PDEGenOrientSolver, PDEPassiveCNSolver):

    """A PDE solver that incorporates crosslink motion through Crank-Nicolson
    integration method"""

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init PDEGenOrientCNSolver ->", end=" ")
        PDEGenOrientSolver.__init__(self, pfile, pdict)
        self.makeDiagMats()
