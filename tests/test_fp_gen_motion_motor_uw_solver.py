#!/usr/bin/env python
import pytest
from foxlink.FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver
from foxlink.FP_gen_orient_motor_UW_solver import FPGenOrientMotorUWSolver
from foxlink.FP_UW_solver import FPUWSolver
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


def test_fp_gen_motion_motor_uw_solver_inheritance():
    """!Test to make sure that FPGenMotionMotorUWSolver is inheriting the correct
    methods from other classes
    @return: void

    """
    assert FPGenMotionMotorUWSolver.ParseParams is FPGenOrientSolver.ParseParams
    assert FPGenMotionMotorUWSolver.makeSolutionGrid is FokkerPlanckSolver.makeSolutionGrid
    assert FPGenMotionMotorUWSolver.setInitialConditions is FokkerPlanckSolver.setInitialConditions
    assert FPGenMotionMotorUWSolver.makeDataframe is FPGenOrientSolver.makeDataframe
    assert FPGenMotionMotorUWSolver.makeDiagMats is FPUWSolver.makeDiagMats

    assert FPGenMotionMotorUWSolver.Run is FokkerPlanckSolver.Run
    assert FPGenMotionMotorUWSolver.Step is not FPGenOrientMotorUWSolver.Step
    assert FPGenMotionMotorUWSolver.stepUW is FPUWSolver.stepUW
    assert FPGenMotionMotorUWSolver.calcForceMatrix is FPGenOrientSolver.calcForceMatrix
    assert FPGenMotionMotorUWSolver.calcTorqueMatrix is FPGenOrientSolver.calcTorqueMatrix
    assert FPGenMotionMotorUWSolver.calcSourceMatrix is FPGenOrientSolver.calcSourceMatrix
    assert FPGenMotionMotorUWSolver.calcVelocityMats is FPGenOrientMotorUWSolver.calcVelocityMats
    assert FPGenMotionMotorUWSolver.RodStep is FPGenMotionSolver.RodStep

    assert FPGenMotionMotorUWSolver.Write is FPGenOrientSolver.Write
    assert FPGenMotionMotorUWSolver.Save is Solver.Save
