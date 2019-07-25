#!/usr/bin/env python
import pytest
from foxlink.FP_gen_motion_solver import FPGenMotionSolver
from foxlink.FP_gen_orient_solver import FPGenOrientSolver
from foxlink.solver import Solver

"""@package docstring
File: test_fp_gen_motion_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@pytest.fixture()
def fp_gen_motion_solver():
    return FPGenMotionSolver()


def test_gen_motion_solver_inheritance():
    """!TODO: Docstring for test_gen_motion_solver.
    @return: TODO

    """
    assert FPGenMotionSolver.Write is FPGenOrientSolver.Write
    assert FPGenMotionSolver.ParseParams is FPGenOrientSolver.ParseParams
    # assert FPGenMotionSolver.Write is not Solver.Write
