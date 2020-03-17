#!/usr/bin/env python
import pytest
from foxlink.pde_gen_motion_solver import PDEGenMotionSolver
from foxlink.pde_gen_orient_solver import PDEGenOrientSolver
from foxlink.rod_motion_solver import RodMotionSolver
from foxlink.pde_solver import PDESolver
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
    assert PDEGenMotionSolver.ParseParams is PDEGenOrientSolver.ParseParams
    assert PDEGenMotionSolver.makeSolutionGrid is PDESolver.makeSolutionGrid
    assert PDEGenMotionSolver.setInitialConditions is PDESolver.setInitialConditions
    assert PDEGenMotionSolver.Run is PDESolver.Run
    assert PDEGenMotionSolver.Step is PDESolver.Step
    assert PDEGenMotionSolver.makeDataframe is PDEGenOrientSolver.makeDataframe
    assert PDEGenMotionSolver.calcForceMatrix is PDEGenOrientSolver.calcForceMatrix
    assert PDEGenMotionSolver.calcTorqueMatrix is PDEGenOrientSolver.calcTorqueMatrix
    assert PDEGenMotionSolver.calcSourceMatrix is PDEGenOrientSolver.calcSourceMatrix
    assert PDEGenMotionSolver.Write is PDEGenOrientSolver.Write
    assert PDEGenMotionSolver.Save is Solver.Save
    assert PDEGenMotionSolver.RodStep is RodMotionSolver.RodStep
