#!/usr/bin/env python
from solver.py import Solver
from scipy.integrate import solve_ivp, dblquad
from .ME_helpers import evolver_zrl
from rod_motion_solver import get_rod_drag_coeff
import numpy as np

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def convert_sol_to_geom(sol):
    return (sol[:3], sol[3:6], sol[6:9], sol[9:12])


def choose_ODE_solver(sol, t, vo, fs, ko, c, ks, beta, L1, L2, d, visc):
    """!Create a closure for ode solver

    @param sol: Array of time-dependent variables in the ODE
    @param t: time
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ko: Turnover rate of motors
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @param d: Diameter of rods
    @param visc: Viscocity of surrounding fluid
    @return: evolver function for ODE of interest

    """
    # Get drag coefficients
    gpara1, gperp1, grot1 = get_rod_drag_coeff(visc, L1, d)
    gpara2, gperp2, grot2 = get_rod_drag_coeff(visc, L2, d)

    def evolver_zrl_closure(sol, t):
        """!Define the function of an ODE solver with certain constant
        parameters.

        @param sol: TODO
        @param t: TODO
        @return: TODO

        """
        r1, r2, u1, u2 = convert_sol_to_geom(sol)
        return evolver_zrl(r1, r2, u1, u2,  # Vectors
                           sol[12], sol[13], sol[14],  # Moments
                           sol[15], sol[16], sol[17],
                           gpara1, gperp1, grot1,  # Friction coefficients
                           gpara2, gperp2, grot2,
                           vo, fs, ko, c, ks, beta, L1, L2)  # Other parameters

    return evolver_zrl_closure


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker - Planck equation
        in a series of moments of motor end positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: TODO
        @param pdict: TODO
        """
        print("Init MomentExpansionSolver -> ")
        Solver.__init__(self, pfile, pdict)

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary
        @return: TODO
        """
        Solver.ParseParams(self)
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
        self.t_eval = np.linspace(0, self.nt, int(self.nt / self.twrite))
        self._nframes = self.t_eval.size

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: TODO
        """
        self.sol = np.zeros(18)
        # Set all geometric variables
        self.sol[:13] = np.concatenate(
            (self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec))
        # TODO Allow for different initial conditions of moments besides zero

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

            self._interaction_grp = self._h5_data.create_group(
                'Interaction_data')

            self.makeXLMomentDataSet()
            self.makeRodDataSet()

            Solver.makeDataframe(self)
            self.data_frame_made = True

    def makeRodDataSet(self):
        """!Initialize dataframe with empty rod configuration data
        @return: void

        """
        self._R1_pos_dset = self._rod_grp.create_dataset(
            'R1_pos', shape=(self._nframes, 3))
        self._R2_pos_dset = self._rod_grp.create_dataset(
            'R2_pos', shape=(self._nframes, 3))
        self._R1_vec_dset = self._rod_grp.create_dataset(
            'R1_vec', shape=(self._nframes, 3))
        self._R2_vec_dset = self._rod_grp.create_dataset(
            'R2_vec', shape=(self._nframes, 3))

    def makeXLMomentDataSet(self):
        """!Initialize dataframe with empty crosslinker moment data
        @return: void

        """
        self._rho_dset = self._xl_grp.create_dataset('zeroth_moment',
                                                     shape=(self._nframes),
                                                     dtype=np.float32)
        self._P_dset = self._xl_grp.create_dataset('first_moments',
                                                   shape=(self._nframes, 2),
                                                   dtype=np.float32)
        self._mu_dset = self._xl_grp.create_dataset('second_moments',
                                                    shape=(self._nframes, 3),
                                                    dtype=np.float32)

    def Run(self):
        """!Run algorithm to solve system of ODEs
        @return: TODO
        """

        self.Save()

    def Write(self):
        """!Write out data
        @return: TODO

        """
        pass
