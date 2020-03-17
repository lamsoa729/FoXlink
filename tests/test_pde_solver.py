#!/usr/bin/env python

import pytest
from foxlink.pde_solver import PDESolver
from foxlink.solver import Solver

"""@package docstring
File: test_fp_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def test_pde_solver_inheritance():
    """!TODO: Docstring for test_gen_motion_solver.
    @return: void

    """
    assert PDESolver.Save is Solver.Save
    # assert PDESolver.ParseParams is FPGenOrientSolver.ParseParams
    # assert PDESolver.makeSolutionGrid is FokkerPlanckSolver.makeSolutionGrid
    # assert PDESolver.setInitialConditions is FokkerPlanckSolver.setInitialConditions
    # assert PDESolver.Run is FokkerPlanckSolver.Run
    # assert PDESolver.Step is FokkerPlanckSolver.Step
    # assert PDESolver.makeDataframe is FPGenOrientSolver.makeDataframe
    # assert PDESolver.calcForceMatrix is FPGenOrientSolver.calcForceMatrix
    # assert PDESolver.calcTorqueMatrix is FPGenOrientSolver.calcTorqueMatrix
    # assert PDESolver.calcSourceMatrix is FPGenOrientSolver.calcSourceMatrix
    # assert PDESolver.Write is FPGenOrientSolver.Write
    # assert PDESolver.RodStep is RodMotionSolver.RodStep


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
