#!/usr/bin/env python
import time
import numpy as np
from scipy.integrate import solve_ivp
from .choose_ME_evolver import choose_ME_evolver
from .solver import Solver

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker - Planck equation
        in a series of moments of motor end positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: TODO
        @param pdict: TODO
        """
        print("Init MomentExpansionSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary
        @return: void
        """
        Solver.ParseParams(self)
        # TODO Non-dimensionalize params right here
        self.R1_pos = np.asarray(self._params['R1_pos'])
        self.R2_pos = np.asarray(self._params['R2_pos'])
        print("R1_pos = ", self.R1_pos)
        print("R2_pos = ", self.R2_pos)
        # Rod orientation vectors
        self.R1_vec = np.asarray(self._params['R1_vec'])
        self.R2_vec = np.asarray(self._params['R2_vec'])
        # Make sure to renormalize
        self.R1_vec /= np.linalg.norm(self.R1_vec)
        self.R2_vec /= np.linalg.norm(self.R2_vec)

        print("R1_vec = ", self.R1_vec)
        print("R2_vec = ", self.R2_vec)

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
        self.t_eval = np.linspace(0, self.nt, int(self.nt / self.twrite) + 1)
        print(self.t_eval)
        self._nframes = self.t_eval.size

        # Set integration method for solver
        if 'method' in self._params:
            self.method = self._params['method']
        else:
            self.method = 'LSODA'
            self._params['method'] = self.method

        # Specify the ODE type
        if 'ODE_type' in self._params:
            self.ODE_type = self._params['ODE_type']
        else:
            self.ODE_type = 'zrl'
            self._params['ODE_type'] = self.ODE_type

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: TODO
        """
        self.sol_init = np.zeros(18)
        # Set all geometric variables
        self.sol_init[:12] = np.concatenate(
            (self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec))
        print("=== Initial conditions ===")
        print(self.sol_init)
        # TODO Allow for different initial conditions of moments besides zero

        # Set solver once you set initial conditions
        self.ode_solver = choose_ME_evolver(self.sol_init,
                                            self._params['vo'],
                                            self._params['fs'],
                                            self._params['ko'],
                                            self._params['co'],
                                            self._params['ks'],
                                            self._params['beta'],
                                            self._params['L1'],
                                            self._params['L2'],
                                            self._params['rod_diameter'],
                                            self._params['viscosity'],
                                            self.ODE_type)

    def makeDataframe(self):
        """!Create data frame to be written out
        @return: TODO
        """
        if not self.data_frame_made:
            self._time_dset = self._h5_data.create_dataset('time',
                                                           data=self.t_eval,
                                                           dtype=np.float32)
            self._xl_grp = self._h5_data.create_group('XL_data')
            self._rod_grp = self._h5_data.create_group('rod_data')

            # self._interaction_grp = self._h5_data.create_group(
            # 'Interaction_data')
            Solver.makeDataframe(self)
            self.data_frame_made = True

    def makeRodDataSet(self):
        """!Initialize dataframe with empty rod configuration data
        @return: void

        """
        self._R1_pos_dset = self._rod_grp.create_dataset(
            'R1_pos', data=self.sol.y[:3, :].T)
        self._R2_pos_dset = self._rod_grp.create_dataset(
            'R2_pos', data=self.sol.y[3:6, :].T)
        self._R1_vec_dset = self._rod_grp.create_dataset(
            'R1_vec', data=self.sol.y[6:9, :].T)
        self._R2_vec_dset = self._rod_grp.create_dataset(
            'R2_vec', data=self.sol.y[9:12, :].T)

    def makeXLMomentDataSet(self):
        """!Initialize dataframe with empty crosslinker moment data
        @return: void

        """
        self._rho_dset = self._xl_grp.create_dataset('zeroth_moment',
                                                     data=self.sol.y[12, :].T,
                                                     dtype=np.float32)
        self._P_dset = self._xl_grp.create_dataset('first_moments',
                                                   data=self.sol.y[13:15, :].T,
                                                   dtype=np.float32)
        self._mu_dset = self._xl_grp.create_dataset('second_moments',
                                                    data=self.sol.y[15:, :].T,
                                                    dtype=np.float32)

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

    def Write(self):
        """!Write out data
        @return: TODO

        """
        self.makeXLMomentDataSet()
        self.makeRodDataSet()
        # Store how long the simulation took
        self._h5_data.attrs['cpu_time'] = self.cpu_time
