#!/usr/bin/env python

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from .choose_ME_evolver import choose_ME_evolver
from .solver import Solver
from .non_dimensionalizer import NonDimensionalizer


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker - Planck equation
        in a series of moments of motor end positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: yaml parameter file name
        @param pdict: parameter dictionary
        """
        print("Init MomentExpansionSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary then calculate
        some necessary parameters if not defined. Also non-dimensionalize parameters.
        @return: void
        """
        Solver.ParseParams(self)

        self.dt = self._params["dt"]  # Time step
        if "nt" not in self._params:
            print("!!! Warning: nt not defined. Using nsteps and dt to find ",
                  "total time. nsteps and dt are not used in this calculation.")
            self.nsteps = int(self._params["nsteps"])
            self.nt = self.nsteps * self.dt
            self._params["nt"] = self.nt
        else:
            self.nt = self._params["nt"]

        if "nwrite" not in self._params:
            self.twrite = self._params["twrite"]
            self.nwrite = int(self.twrite / self.dt)
        elif "twrite" not in self._params:
            self.nwrite = self._params["nwrite"]
            self.twrite = float(self.nwrite * self.dt)
        else:
            print("!!! Warning: Write parameters over defined,",
                  "using twrite to calculate number of steps between write out.")
            self.twrite = self._params["twrite"]
            self.nwrite = int(self.twrite / self.dt)

        self.non_dimmer = self.non_dimensionalize()
        # Rod orientation vectors
        self.R1_vec = np.asarray(self._params['R1_vec'])
        self.R2_vec = np.asarray(self._params['R2_vec'])
        # Make sure to renormalize
        self.R1_vec /= np.linalg.norm(self.R1_vec)
        self.R2_vec /= np.linalg.norm(self.R2_vec)

        print("R1_vec = ", self.R1_vec)
        print("R2_vec = ", self.R2_vec)

        self.t_eval = np.linspace(0, self.nt, int(self.nt / self.twrite) + 1)
        self._nframes = self.t_eval.size

        # Set integration method for solver
        if 'method' in self._params:
            self.method = self._params['method']
        else:
            self.method = 'LSODA'
            self._params['method'] = self.method
        print(self.method)

        # Specify the ODE type
        if 'ODE_type' in self._params:
            self.ODE_type = self._params['ODE_type']
        else:
            self.ODE_type = 'zrl'
            self._params['ODE_type'] = self.ODE_type

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: void
        """
        self.sol_init = np.zeros(18)
        # Set all geometric variables
        self.sol_init[:12] = np.concatenate(
            (self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec))
        print("=== Initial conditions ===")
        print(self.sol_init)
        # TODO Allow for different initial conditions of moments besides zero

        # Set solver once you set initial conditions
        # Add kwargs
        self.ode_solver = choose_ME_evolver(self.sol_init, self)

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
        self.cpu_time = time.time() - t0
        print(
            r" --- Total simulation time {:.4f} seconds ---".format(self.cpu_time))

        self.Write()

    def makeRodDataSet(self):
        """!Initialize dataframe with empty rod configuration data
        @return: void

        """
        self._R1_pos_dset = self._rod_grp.create_dataset(
            'R1_pos', data=self.sol.y[: 3, :].T)
        self._R2_pos_dset = self._rod_grp.create_dataset(
            'R2_pos', data=self.sol.y[3: 6, :].T)
        self._R1_vec_dset = self._rod_grp.create_dataset(
            'R1_vec', data=self.sol.y[6: 9, :].T)
        self._R2_vec_dset = self._rod_grp.create_dataset(
            'R2_vec', data=self.sol.y[9: 12, :].T)

    def makeXLMomentDataSet(self):
        """!Initialize dataframe with empty crosslinker moment data
        @return: void

        """
        self._rho_dset = self._xl_grp.create_dataset('zeroth_moment',
                                                     data=self.sol.y[12, :].T,
                                                     dtype=np.float32)
        self._P_dset = self._xl_grp.create_dataset('first_moments',
                                                   data=self.sol.y[13: 15, :].T,
                                                   dtype=np.float32)
        self._mu_dset = self._xl_grp.create_dataset('second_moments',
                                                    data=self.sol.y[15:, :].T,
                                                    dtype=np.float32)

    def Write(self):
        """!Write out data
        @return: void

        """
        self.redimensionalize()
        self.makeXLMomentDataSet()
        self.makeRodDataSet()
        # Store how long the simulation took
        self._h5_data.attrs['cpu_time'] = self.cpu_time

    def non_dimensionalize(self):
        """!Non-dimensionalize parameters to reduce error in calculations.
        @return: non dimensionalizer

        """
        # non_dim_dict = {'time': 1. / self._params['ko'],
        #                 # 'length': max(self._params['L1'], self._params['L2']),
        #                 'length': self._params['fs'] / self._params['ks'],
        #                 'energy': 1. / self._params['beta']}
        non_dim_dict = {'time': 1.,
                        # 'length': max(self._params['L1'], self._params['L2']),
                        'length': 1.,
                        'energy': 1.}
        non_dimmer = NonDimensionalizer(**non_dim_dict)
        # non_dimmer.calc_new_dim('force', ['energy', 'length'], [1, -1])

        self.beta = non_dimmer.non_dim_val(self._params['beta'],
                                           ['energy'], [-1])
        self.visc = non_dimmer.non_dim_val(self._params['viscosity'],
                                           ['energy', 'time', 'length'],
                                           [1, 1, -3])
        self.L_i = non_dimmer.non_dim_val(self._params['L1'], ['length'])
        self.L_j = non_dimmer.non_dim_val(self._params['L2'], ['length'])
        self.R1_pos = non_dimmer.non_dim_val(
            self._params['R1_pos'], ['length'])
        self.R2_pos = non_dimmer.non_dim_val(
            self._params['R2_pos'], ['length'])
        self.rod_diam = non_dimmer.non_dim_val(self._params['rod_diameter'],
                                               ['length'])
        self.dt = non_dimmer.non_dim_val(self.dt, ['time'])
        self.nt = non_dimmer.non_dim_val(self.nt, ['time'])
        self.twrite = non_dimmer.non_dim_val(self.twrite, ['time'])
        self.ko = non_dimmer.non_dim_val(self._params['ko'], ['time'], [-1])
        self.co = non_dimmer.non_dim_val(self._params['co'], ['length'], [-2])
        self.ks = non_dimmer.non_dim_val(self._params['ks'],
                                         ['energy', 'length'], [1, -2])
        self.ho = non_dimmer.non_dim_val(self._params['ho'], ['length'])
        self.vo = non_dimmer.non_dim_val(self._params['vo'],
                                         ['length', 'time'], [1, -1])
        self.fs = non_dimmer.non_dim_val(
            self._params['fs'], ['energy', 'length'], [1, -1])
        return non_dimmer

    def redimensionalize(self):
        """!Redimensionalize data arrays
        @return: void

        """
        # Redimensionalize rod positions
        self.sol.y[: 6, :] = self.non_dimmer.dim_val(
            self.sol.y[: 6, :], ['length'])
        # Redimensionalize first moments
        self.sol.y[13: 15, :] = self.non_dimmer.dim_val(self.sol.y[13: 15, :],
                                                        ['length'])
        # Redimensionalize second moments
        self.sol.y[15:, :] = self.non_dimmer.dim_val(self.sol.y[15:, :],
                                                     ['length'], [2])
