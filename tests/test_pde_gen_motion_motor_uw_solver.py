#!/usr/bin/env python
import pytest
from foxlink.pde_gen_motion_motor_uw_solver import PDEGenMotionMotorUWSolver
from foxlink.pde_gen_orient_motor_uw_solver import PDEGenOrientMotorUWSolver
from foxlink.pde_uw_solver import PDEUWSolver
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


def test_fp_gen_motion_motor_uw_solver_inheritance():
    """!Test to make sure that PDEGenMotionMotorUWSolver is inheriting the correct
    methods from other classes
    @return: void

    """
    assert PDEGenMotionMotorUWSolver.ParseParams is PDEGenOrientSolver.ParseParams
    assert PDEGenMotionMotorUWSolver.makeSolutionGrid is PDESolver.makeSolutionGrid
    assert PDEGenMotionMotorUWSolver.setInitialConditions is PDESolver.setInitialConditions
    assert PDEGenMotionMotorUWSolver.makeDataframe is PDEGenOrientSolver.makeDataframe
    assert PDEGenMotionMotorUWSolver.makeDiagMats is PDEUWSolver.makeDiagMats

    assert PDEGenMotionMotorUWSolver.Run is PDESolver.Run
    assert PDEGenMotionMotorUWSolver.Step is not PDEGenOrientMotorUWSolver.Step
    assert PDEGenMotionMotorUWSolver.stepUW is PDEUWSolver.stepUW
    assert PDEGenMotionMotorUWSolver.calcForceMatrix is PDEGenOrientSolver.calcForceMatrix
    assert PDEGenMotionMotorUWSolver.calcTorqueMatrix is PDEGenOrientSolver.calcTorqueMatrix
    assert PDEGenMotionMotorUWSolver.calcSourceMatrix is PDEGenOrientSolver.calcSourceMatrix
    assert PDEGenMotionMotorUWSolver.calcVelocityMats is PDEGenOrientMotorUWSolver.calcVelocityMats
    assert PDEGenMotionMotorUWSolver.RodStep is PDEGenMotionSolver.RodStep

    assert PDEGenMotionMotorUWSolver.Write is PDEGenOrientSolver.Write
    assert PDEGenMotionMotorUWSolver.Save is Solver.Save
