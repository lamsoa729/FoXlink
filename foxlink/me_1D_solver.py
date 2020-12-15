#!/usr/bin/env python

"""@package docstring
File: me_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from .create_me_1D_evolver import create_me_1D_evolver
from .solver import Solver
from .non_dimensionalizer import NonDimensionalizer


class MomentExpansion1DSolver(Solver):

    """! Find the average moments of motor densities between filaments bundles
    in 1-dimension.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: yaml parameter file name
        @param pdict: parameter dictionary
        """
        print("Init MomentExpansion1DAvgSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary then calculate
        some necessary parameters if not defined.
        Also non-dimensionalize parameters.
        @return: void
        """
        Solver.ParseParams(self)

        self.nt = self._params["nt"]
        self.twrite = self._params["twrite"]
        # self.nwrite = int(self.twrite / self.dt)

        self.non_dimmer = self.non_dimensionalize()

        self.t_eval = np.linspace(0, self.nt, int(self.nt / self.twrite) + 1)
        self._nframes = self.t_eval.size

        # Set integration method for solver
        self.method = self._params.get('method', 'BDF')
        self._params['method'] = self.method
        print("Solving method = ", self.method)

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: void
        """
        self.sol_init = np.zeros(4)
        # Set all geometric variables
        self.sol_init[0] = self.xi_pos
        print("=== Initial conditions ===")
        print(self.sol_init)

        # Set solver once you set initial conditions
        self.ode_solver = create_me_1D_evolver(self, self.sol_init)

    def makeDataframe(self):
        """!Create data frame to be written out
        @return: TODO
        """
        t_arr = self.non_dimmer.dim_val(self.t_eval, ['time'])
        print("Evaluated at:", t_arr)
        if not self.data_frame_made:
            self._time_dset = self._h5_data.create_dataset('time',
                                                           data=t_arr,
                                                           dtype=np.float32)
            self._xl_grp = self._h5_data.create_group('xl_data')
            self._rod_grp = self._h5_data.create_group('rod_data')

            Solver.makeDataframe(self)
            self.data_frame_made = True

    def Run(self):
        """!Run algorithm to solve system of ODEs
        @return: TODO
        """

        t0 = time.time()
        self.sol = solve_ivp(self.ode_solver, [0, self.nt], self.sol_init,
                             t_eval=self.t_eval, method=self.method)
        # min_step=self.dt, atol=1e-6)
        self.cpu_time = time.time() - t0
        print(
            r" --- Total simulation time {:.4f} seconds ---".format(
                self.cpu_time))

        self.Write()

    def make_rod_dataset(self):
        """!Initialize dataframe with empty rod configuration data
        @return: void

        """
        self._xi_pos_dset = self._rod_grp.create_dataset(
            'xi_pos', data=self.sol.y[0, :].T)

    def make_xl_moment_dataset(self):
        """!Initialize dataframe with empty crosslinker moment data
        Nomenclature:
            hat{x} = ->

            rod bundle L <------------> R
                  <-----                ----->  if rod i starts here,
            if rod i starts here                it moments are labeled muRP
            its moments are labeled muLN
        moments are in order of mu00, mu10, mu01
        @return: void

        """

        self._mu_dset = self._xl_grp.create_dataset(
            'mu_moments', data=self.sol.y[1:, :].T, dtype=np.float32)

    def Write(self):
        """!Write out data
        @return: void

        """
        self.redimensionalize()
        self.make_xl_moment_dataset()
        self.make_rod_dataset()
        # Store how long the simulation took
        self._h5_data.attrs['cpu_time'] = self.cpu_time

    def non_dimensionalize(self):
        """!Non-dimensionalize parameters to reduce error in calculations.
        @return: non dimensionalizer

        """
        non_dim_dict = {'time': 1.,
                        # 'length': float(self._params['L']),
                        'length': 1.,
                        'energy': 1.}
        non_dimmer = NonDimensionalizer(**non_dim_dict)
        # non_dimmer.calc_new_dim('force', ['energy', 'length'], [1, -1])

        self.beta = non_dimmer.non_dim_val(self._params['beta'],
                                           ['energy'], [-1])
        self.visc = non_dimmer.non_dim_val(self._params['viscosity'],
                                           ['energy', 'time', 'length'],
                                           [1, 1, -3])
        self.xi_pos = non_dimmer.non_dim_val(
            self._params['xi_pos'], ['length'])
        self.L = non_dimmer.non_dim_val(self._params['L'], ['length'])
        self.rod_diam = non_dimmer.non_dim_val(self._params['rod_diameter'],
                                               ['length'])
        self.nt = non_dimmer.non_dim_val(self.nt, ['time'])
        self.twrite = non_dimmer.non_dim_val(self.twrite, ['time'])
        self.ko = non_dimmer.non_dim_val(self._params['ko'], ['time'], [-1])
        self.co = non_dimmer.non_dim_val(self._params['co'], ['length'], [-2])
        self.ks = non_dimmer.non_dim_val(self._params['ks'],
                                         ['energy', 'length'], [1, -2])
        self.vo = non_dimmer.non_dim_val(self._params['vo'],
                                         ['length', 'time'], [1, -1])
        self.fs = non_dimmer.non_dim_val(
            self._params['fs'], ['energy', 'length'], [1, -1])
        self.ui = float(self._params['ui'])
        self.ui /= abs(self.ui)
        self.uj = float(self._params['uj'])
        self.uj /= abs(self.uj)
        return non_dimmer

    def redimensionalize(self):
        """!Redimensionalize data arrays
        @return: void

        """
        # Redimensionalize rod positions
        self.sol.y[0, :] = self.non_dimmer.dim_val(self.sol.y[0, :],
                                                   ['length'])
        # Redimensionalize first moments
        self.sol.y[2:, :] = self.non_dimmer.dim_val(self.sol.y[2:, :],
                                                    ['length'])
