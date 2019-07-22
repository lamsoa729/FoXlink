#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
from .FP_gen_orient_solver import FPGenOrientSolver
from .FP_static_xlinks_solver import FPStaticXlinksSolver


"""@package docstring
File: FP_gen_orient_static_xlinks.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPGenOrientStaticXlinksSolver(FPGenOrientSolver, FPStaticXlinksSolver):

    """Solver class to calculate static crosslinks binding and unbinding from solution."""

    def __init__(self, pfile=None, pdict=None):
        """Initialize class in a general orientation"""
        print("Init FPGenOrientStaticXlinks ->", end=" ")
        FPGenOrientSolver.__init__(self, pfile=pfile, pdict=pdict)
