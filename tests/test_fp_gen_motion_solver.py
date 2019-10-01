#!/usr/bin/env python
import pytest
from foxlink.FP_gen_motion_solver import FPGenMotionSolver
from foxlink.FP_gen_orient_solver import FPGenOrientSolver
from foxlink.rod_motion_solver import RodMotionSolver
from foxlink.FP_solver import FokkerPlanckSolver
from foxlink.solver import Solver

"""@package docstring
File: test_fp_gen_motion_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def test_gen_motion_solver_inheritance():
    """!TODO: Docstring for test_gen_motion_solver.
    @return: void

    """
    assert FPGenMotionSolver.ParseParams is FPGenOrientSolver.ParseParams
    assert FPGenMotionSolver.makeSolutionGrid is FokkerPlanckSolver.makeSolutionGrid
    assert FPGenMotionSolver.setInitialConditions is FokkerPlanckSolver.setInitialConditions
    assert FPGenMotionSolver.Run is FokkerPlanckSolver.Run
    assert FPGenMotionSolver.Step is FokkerPlanckSolver.Step
    assert FPGenMotionSolver.makeDataframe is FPGenOrientSolver.makeDataframe
    assert FPGenMotionSolver.calcForceMatrix is FPGenOrientSolver.calcForceMatrix
    assert FPGenMotionSolver.calcTorqueMatrix is FPGenOrientSolver.calcTorqueMatrix
    assert FPGenMotionSolver.calcSourceMatrix is FPGenOrientSolver.calcSourceMatrix
    assert FPGenMotionSolver.Write is FPGenOrientSolver.Write
    assert FPGenMotionSolver.Save is Solver.Save
    assert FPGenMotionSolver.RodStep is RodMotionSolver.RodStep
