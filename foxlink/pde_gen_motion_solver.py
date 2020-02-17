#!/usr/bin/env python
from .pde_gen_orient_solver import PDEGenOrientSolver
from .rod_motion_solver import RodMotionSolver


"""@package docstring
File: pde_gen_motion_Solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Combination class of general orientation solver and rod motion solver
"""


class PDEGenMotionSolver(PDEGenOrientSolver, RodMotionSolver):

    """Docstring for PDEGenMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init PDEGenMotionSolver ->", end=" ")
        PDEGenOrientSolver.__init__(self, pfile=pfile, pdict=pdict)
