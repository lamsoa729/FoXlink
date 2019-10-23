#!/usr/bin/env python
from scipy.integrate import solve_ivp, dblquad
import time
import numpy as np
from .ME_helpers import (evolver_zrl, evolver_zrl_stat,
                         boltz_fact_zrl, weighted_boltz_fact_zrl)
from .rod_motion_solver import get_rod_drag_coeff
from .solver import Solver

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def convert_sol_to_geom(sol):
    return (sol[:3], sol[3:6], sol[6:9], sol[9:12])


def sol_print_out(sol):
    """!Print out current solution to solver

    @param r1: Center of mass postion of rod1
    @param r2: Center of mass position of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param sol: Full solution array of ODE
    @return: void

    """
    r1, r2, u1, u2 = convert_sol_to_geom(sol)
    print ("Step-> r1:", r1, ", r2:", r2, ", u1:", u1, ", u2:", u2)
    print ("       rho:{}, P1:{}, P2:{}, mu11:{}, mu20:{}, mu02:{}".format(
        sol[12], sol[13], sol[14],
        sol[15], sol[16], sol[17]))


def prep_zrl_stat_evolver(sol, ks, beta, L1, L2):
    """!TODO: Docstring for prep_zrl_stat_evolver.

    @param arg1: TODO
    @return: TODO

    """
    r1, r2, u1, u2 = convert_sol_to_geom(sol)
    r12 = r2 - r1
    rsqr = np.dot(r12, r12)
    a1 = np.dot(r12, u1)
    a2 = np.dot(r12, u2)
    b = np.dot(u1, u2)
    # TODO computer source terms
    q, e = dblquad(boltz_fact_zrl, -.5 * L1, .5 * L1,
                   lambda s2: -.5 * L2, lambda s2: .5 * L2,
                   args=[rsqr, a1, a2, b, ks, beta])
    q10, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                     lambda s2: -.5 * L2, lambda s2: .5 * L2,
                     args=[1, 0, rsqr, a1, a2, b, ks, beta],)
    q01, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                     lambda s2: -.5 * L2, lambda s2: .5 * L2,
                     args=[0, 1, rsqr, a1, a2, b, ks, beta])
    q11, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                     lambda s2: -.5 * L2, lambda s2: .5 * L2,
                     args=[1, 1, rsqr, a1, a2, b, ks, beta])
    q20, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                     lambda s2: -.5 * L2, lambda s2: .5 * L2,
                     args=[2, 0, rsqr, a1, a2, b, ks, beta])
    q02, e = dblquad(weighted_boltz_fact_zrl, -.5 * L1, .5 * L1,
                     lambda s2: -.5 * L2, lambda s2: .5 * L2,
                     args=[0, 2, rsqr, a1, a2, b, ks, beta])
    return rsqr, a1, a2, b, q, q10, q01, q11, q20, q02


def choose_ODE_solver(sol, vo, fs, ko, c, ks, beta, L1,
                      L2, d, visc, ODE_type='zrl'):
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
    @param ODE_type: Which ODE to use (zrl, zrl_stat)
    @return: evolver function for ODE of interest

    """

    if ODE_type == 'zrl':
        # Get drag coefficients
        gpara1, gperp1, grot1 = get_rod_drag_coeff(visc, L1, d)
        gpara2, gperp2, grot2 = get_rod_drag_coeff(visc, L2, d)

        def evolver_zrl_closure(t, sol):
            """!Define the function of an ODE solver with zero length
            crosslinking proteins and moving rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl

            """
            if not np.all(np.isfinite(sol)):
                raise RuntimeError(
                    'Infinity or NaN thrown in ODE solver solutions. Current solution', sol)

            r1, r2, u1, u2 = convert_sol_to_geom(sol)
            # sol_print_out(sol)
            return evolver_zrl(r1, r2, u1, u2,  # Vectors
                               sol[12], sol[13], sol[14],  # Moments
                               sol[15], sol[16], sol[17],
                               gpara1, gperp1, grot1,  # Friction coefficients
                               gpara2, gperp2, grot2,
                               vo, fs, ko, c, ks, beta, L1, L2, fast='fast')  # Other parameters
        return evolver_zrl_closure

    elif ODE_type == 'zrl_stat':
        # Compute geometric terms that will not change
        rsqr, a1, a2, b, q, q10, q01, q11, q20, q02 = prep_zrl_stat_evolver(
            sol, ks, beta, L1, L2)

        def evolver_zrl_stat_closure(t, sol):
            """!Define the function of an ODE solver with zero rest length
            crosslinking protiens and stationary rods.

            @param t: Time array
            @param sol: Solution array
            @return: Function to ODE zrl stat

            """
            # sol_print_out(sol)
            return evolver_zrl_stat(sol[12], sol[13], sol[14],  # Moments
                                    sol[15], sol[16], sol[17],
                                    rsqr, a1, a2, b, q, q10, q01, q11, q20, q02,  # Pre-computed values
                                    vo, fs, ko, c, ks, beta, L1, L2)  # Other parameters
        return evolver_zrl_stat_closure

    else:
        raise IOError('{} not a defined ODE equation for foxlink.')


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker - Planck equation
        in a series of moments of motor end positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: TODO
        @param pdict: TODO
        """
        print("Init MomentExpansionSolver -> ", end=" ")
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
        self.ode_solver = choose_ODE_solver(self.sol_init,
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
