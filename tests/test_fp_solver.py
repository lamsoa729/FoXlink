#!/usr/bin/env python

import pytest
from foxlink.FP_solver import FokkerPlanckSolver
from foxlink.solver import Solver

"""@package docstring
File: test_fp_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def test_fp_solver_inheritance():
    """!TODO: Docstring for test_gen_motion_solver.
    @return: void

    """
    assert FokkerPlanckSolver.Save is Solver.Save
    # assert FokkerPlanckSolver.ParseParams is FPGenOrientSolver.ParseParams
    # assert FokkerPlanckSolver.makeSolutionGrid is FokkerPlanckSolver.makeSolutionGrid
    # assert FokkerPlanckSolver.setInitialConditions is FokkerPlanckSolver.setInitialConditions
    # assert FokkerPlanckSolver.Run is FokkerPlanckSolver.Run
    # assert FokkerPlanckSolver.Step is FokkerPlanckSolver.Step
    # assert FokkerPlanckSolver.makeDataframe is FPGenOrientSolver.makeDataframe
    # assert FokkerPlanckSolver.calcForceMatrix is FPGenOrientSolver.calcForceMatrix
    # assert FokkerPlanckSolver.calcTorqueMatrix is FPGenOrientSolver.calcTorqueMatrix
    # assert FokkerPlanckSolver.calcSourceMatrix is FPGenOrientSolver.calcSourceMatrix
    # assert FokkerPlanckSolver.Write is FPGenOrientSolver.Write
    # assert FokkerPlanckSolver.RodStep is RodMotionSolver.RodStep


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
