#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
from .me_solver import MomentExpansionSolver
from .non_dimensionalizer import NonDimensionalizer

# Initial conditions for the system
xlink_number = 100


# Build vector


class NFilMomentExpansionSolver(MomentExpansionSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: yaml parameter file name
        @param pdict: parameter dictionary
        """
        print("Init MomentExpansionSolver ->", end=" ")
        MomentExpansionSolver.__init__(self, pfile, pdict)

    def set_rod_params(self):
        # go through parameter file pulling out rod parameters
        print("Set rod parameters")
        pass

    def setInitialConditions(self):
        """! Set the initial state for the solution.

        @return: void, modifies solution grid

        """
        print("Setting initial conditions")
        pass

    def Run(self):
        print("Run MomentExpansionSolver")

    def make_rod_dataset(self):
        pass

    def make_xl_moment_dataset(self):
        pass

    def non_dimensionalize(self):
        """!Non-dimensionalize parameters to reduce error in calculations.
        @return: non dimensionalizer

        """
        # FIXME: Fix the length non-dimensionalization
        non_dim_dict = {
            "time": 1.0,
            "length": 1.0,
            # "length": float(max(self._params["L1"], self._params["L2"])),
            "energy": 1.0,
        }
        non_dimmer = NonDimensionalizer(**non_dim_dict)
        # non_dimmer.calc_new_dim('force', ['energy', 'length'], [1, -1])

        self.beta = non_dimmer.non_dim_val(self._params["beta"], ["energy"], [-1])
        self.visc = non_dimmer.non_dim_val(
            self._params["viscosity"], ["energy", "time", "length"], [1, 1, -3]
        )
        self.rod_diam = non_dimmer.non_dim_val(self._params["rod_diameter"], ["length"])
        self.dt = non_dimmer.non_dim_val(self.dt, ["time"])
        self.nt = non_dimmer.non_dim_val(self.nt, ["time"])
        self.twrite = non_dimmer.non_dim_val(self.twrite, ["time"])
        self.ko = non_dimmer.non_dim_val(self._params["ko"], ["time"], [-1])
        self.ks = non_dimmer.non_dim_val(
            self._params["ks"], ["energy", "length"], [1, -2]
        )
        return non_dimmer

    def redimensionalize(self):
        pass
