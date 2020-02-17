#!/usr/bin/env python
"""@package docstring
File: pde_gen_orient_static_xlinks.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

from .pde_gen_orient_solver import PDEGenOrientSolver
from .pde_static_xlinks_solver import PDEStaticXlinksSolver


class PDEGenOrientStaticXlinksSolver(
        PDEGenOrientSolver, PDEStaticXlinksSolver):

    """Solver class to calculate static crosslinks binding and unbinding from solution."""

    def __init__(self, pfile=None, pdict=None):
        """Initialize class in a general orientation"""
        print("Init PDEGenOrientStaticXlinks ->", end=" ")
        PDEGenOrientSolver.__init__(self, pfile=pfile, pdict=pdict)
