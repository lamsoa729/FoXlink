# -*- coding: utf-8 -*-
"""Tests for `foxlink` package."""

import pytest

from foxlink import foxlink
from foxlink.solver import Solver
from pathlib import Path


def test_solver_init():
    def_params = Solver.default_params
    svlr = Solver()
    assert svlr._params == def_params
    # Clean up
    svlr._h5_fpath.unlink()
