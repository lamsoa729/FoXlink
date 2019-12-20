#!/usr/bin/env python

"""@package docstring
File: solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Base FokkerPlanckSolver class for FoXlink
"""

import time
import numpy as np
from scipy import sparse
from .solver import Solver


class FokkerPlanckSolver(Solver):

    """!Abstract class for solver objects. All PDE algorithms are implemented
    through these classes.

        Common method functions for inherited solver classes:
            Run = Iterate through time using the Step method
            Step <- Abstract = Implement the specified algorithm at a times step
            ParseParams = Parse parameters from file that is given
            setInitialConditions = Set the initial stat of simulation
            makeSolutionGrid = Create data structures that hold information
                               about the position of xlinks and rods.
            calc<>Matrix(ces) <- Abstract = Calculate current data structures
                                            used to evolve system in time.
            Write = At a specified time step add current solution to output file.
            Save = Flush the remaining data in hdf5 file.
        Foxlink uses inheritance to quickly create and combine PDE solving
        algorithms. While solvers do not need to follow specific guidelines, it
        is useful to categorize solvers into different types avoid inheritance
        conflicts when using multi-inheritance. These include:
            Orientation solver -> Defines rods orientations and associate parameters.
                                  eg. Parallel (para),
            Xlink algorithm solver -> What technique is used to solve the xlink
                                      density equations. Can use multiple in one
                                      system. eg. Upwind (UW), Crank-Nicolson(CN)
            Xlink type solver -> What type of crosslinks are being modeled.
                                 eg. Motor, Static, Passive
            Rod algorithm solver -> What technique is used to solve the rods
                                      position equations.
                                      eg. Orient (no motions), Motion, Optical Trap

        Derivative class names are based on their solver type.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions

        @param pfile: parameter file for PDEs

        """
        print("Init FokkerPlanckSolver ->", end=" ")
        Solver.__init__(self, pfile, pdict)

    def ParseParams(self):
        """! Method to extract and/or calculate parameter values necessary for
        solving algorithms to run.

        @param skip: Load parameter file but skip rest of the parsing. Used
                     when you need to load special parameters before making
                     solution grid in subclasses.

        @return: void, modifies self._params

        """
        Solver.ParseParams(self)

        # Integration parameters
        self.t = 0.
        self.ds = self._params["ds"]  # Segmentation size of microtubules
        if "nt" not in self._params:
            self.nsteps = int(self._params["nsteps"])
            self.dt = self._params["dt"]  # Time step
            self.nt = self.nsteps * self.dt
            self._params["nt"] = self.nt
        elif "dt" not in self._params:
            self.nt = self._params["nt"]  # total time
            self.nsteps = self._params["nsteps"]
            self.dt = float(self.nt / self.nsteps)
            self._params["dt"] = self.dt
        elif "nsteps" not in self._params:
            self.nt = self._params["nt"]  # total time
            self.dt = self._params["dt"]  # Time step
            self.nsteps = int(self.nt / self.dt)
            self._params["nsteps"] = self.nsteps
        else:
            print("!!! Warning: step parameters over defined,",
                  "using nt and nsteps to calculate step size.")
            self.nt = self._params["nt"]  # total time
            self.dt = self._params["dt"]  # Time step
            self.nsteps = int(self.nt / self.dt)
            self._params["nsteps"] = self.nsteps

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
        # Make time array. Set extra space for initial condition
        self.time = np.linspace(0, self.nt, self.nsteps + 1)
        self._nframes = len(self.time[::self.nwrite])
        print("Time step: ", self.dt)
        print("Total time: ", self.nt)
        print("Number of steps: ", self.nsteps)
        print("Write out every {} steps ({} secs)".format(self.nwrite,
                                                          self.twrite))

    def setInitialConditions(self):
        """! Set the initial state for the solution grid.
            If no 'initial_condition' parameter is set in the yaml file or
            parameter dictionary. The solution grid will remain filled with zeros.

        @return: void, modifies solution grid

        """
        self.makeSolutionGrid()
        if 'initial_condition' in self._params:
            if self._params['initial_condition'] == 'equil':
                self.sgrid += self.src_mat / self._params['ko']
            elif self._params['initial_condition'] == 'empty':
                return
            else:
                self.sgrid += eval(self._params['initial_condition'])
            print("Initial distribution =", self.sgrid)

    def makeSolutionGrid(self):
        """!Make an array of solutions to solve the PDE
        @return: void

        """
        L1 = self._params["L1"]
        L2 = self._params["L2"]

        ds = self.ds
        self.ns1 = int(L1 / ds) + 2
        self.ns2 = int(L2 / ds) + 2

        # FIXME This maintains proper spacing but I should check this with
        #       someone who understands methods better
        self._params["L1"] = ds * (self.ns1 - 2)
        self._params["L2"] = ds * (self.ns2 - 2)

        # Discrete rod locations, extra spots left for boundary conditions
        self.s1, step1 = np.linspace(
            0, ds * (self.ns1 - 1), self.ns1, retstep=True)
        self.s1 -= (L1 * .5)
        self.s2, step2 = np.linspace(
            0, ds * (self.ns2 - 1), self.ns2, retstep=True)
        self.s2 -= (L2 * .5)
        print("ds1: ", step1)
        print("ds2: ", step2)

        # self.sgrid = sparse.csc_matrix((self.ns1, self.ns2))
        self.sgrid = np.zeros((self.ns1, self.ns2))
        self.calcSourceMatrix()
        self.calcInteractions()

    def calcInteractions(self):
        """!TODO: Docstring for CalcInteractions.
        @return: TODO

        """
        self.calcForceMatrix()
        self.calcTorqueMatrix()
        self.cleared = False

    def Run(self):
        """!Run PDE solver with parameters in pfile through explicity interative time stepping.
        @return: void

        """
        # Write initial configuration
        self.Write()
        self.written = False

        t0 = time.time()
        t_start = t0
        while self.t < self.nt:
            self.Step()
            self.t += self.dt
            # Get rid of subnormal numbers for speed
            self.sgrid = self.sgrid.round(30)
            if (int(self.t / self.dt) % self.nwrite) == 0:
                t1 = time.time()
                print(r" {} steps in {:.4f} seconds, {:.1f}% complete".format(
                    self.nwrite, t1 - t0, float(self.t / self.nt) * 100.))
                self.Write()
                # Reset write function
                self.written = False
                t0 = time.time()
            # Clear interactions for next step
            self.clearInteractions()
        tot_time = time.time() - t_start
        print(r" --- Total of {:d} steps in {:.4f} seconds ---".format(self.nsteps,
                                                                       tot_time))
        self._h5_data.attrs['run_time'] = tot_time

        return

    def Step(self):
        """!Step solver method one unit in time
        @return: Sum of changes of grid, Changes phi1 and phi0

        """
        print("Step not made!",
              "Initialize Step method of {}.".format(self.__class__.__name__))

    def makeDataframe(self):
        """! Make output data frame
        @return: void

        """
        if not self.data_frame_made:
            time = self.time[::self.nwrite]
            self._time_dset = self._h5_data.create_dataset('time', data=time,
                                                           dtype=np.float32)
            self._xl_grp = self._h5_data.create_group('XL_data')
            self._rod_grp = self._h5_data.create_group('rod_data')

            self._rod_grp.create_dataset('s1', data=self.s1)
            self._rod_grp.create_dataset('s2', data=self.s2)

            self._interaction_grp = self._h5_data.create_group(
                'Interaction_data')

            self.makeXLDataSet()
            self.makeInteractionDataSet()
            Solver.makeDataframe(self)
            self.data_frame_made = True

    def makeXLDataSet(self):
        """!Make specific data set for crosslinking proteins
        @return: void

        """
        self._xl_distr_dset = self._xl_grp.create_dataset(
            'XL_distr',
            shape=(self.ns1, self.ns2, self._nframes),
            dtype=np.float32)

    def makeInteractionDataSet(self):
        """!Make specific data set for interaction data
        @return: TODO

        """
        self._force_dset = self._interaction_grp.create_dataset(
            'force_data',
            shape=(self._nframes, 2, 3),
            dtype=np.float32)
        for dim, label in zip(self._force_dset.dims,
                              ['frame', 'rod', 'coord']):
            dim.label = label
        self._torque_dset = self._interaction_grp.create_dataset(
            'torque_data',
            shape=(self._nframes, 2, 3),
            dtype=np.float32)
        for dim, label in zip(self._torque_dset.dims,
                              ['frame', 'rod', 'coord']):
            dim.label = label

    def calcSourceMatrix(self):
        """Virtual functions for calculating source matrix
        @return: void, modifies self.src_mat

        """
        print("calcSourceMatrix not implemented. Source matrix initialized with zeros.")
        self.src_mat = sparse.csc_matrix((self.ns1, self.ns2))

    def calcForceMatrix(self):
        """Virtual functions for calculating force matrix for a given configuration
        @return: void, modifies self.f_mat

        """
        print("calcforceMatrix not implemented. Force matrix initialized with zeros.")
        self.f_mat = np.zeros((self.ns1, self.ns2, 3))
        # self.f_mat = sparse.csc_matrix((self.ns1, self.ns2, 3))

    def calcTorqueMatrix(self):
        """Virtual functions for calculating force matrix for a given configuration
        @return: void, modifies self.t_mat

        """
        print("calcTorqueMatrix not implemented. Torque matrix initialized with zeros.")
        self.t_mat = np.zeros((self.ns1, self.ns2, 3))

    def clearInteractions(self):
        """!TODO: Docstring for clearInteractions.
        @return: TODO

        """
        self.force1 = 0
        self.force2 = 0
        self.torque1 = 0
        self.torque2 = 0
        self.cleared = True

    def apply_dirichlet_bc(self, bc_val=0.):
        """! Apply a constant boundary condition to solution grid
        @return: TODO

        """
        self.sgrid[:, 0] *= bc_val
        self.sgrid[:, -1] *= bc_val
        self.sgrid[0, :] *= bc_val
        self.sgrid[-1, :] *= bc_val

    def Write(self):
        """!Write current step in algorithm into data frame
        @return: index of current step

        """
        i_step = int((self.t / self.dt) / self.nwrite)
        if not self.written:
            self._xl_distr_dset[:, :, i_step] = self.sgrid
            self._time_dset[i_step] = self.t
            if self.cleared:
                self.calcInteractions()
            self._force_dset[i_step, 0] = self.force1
            self._force_dset[i_step, 1] = self.force2
            self._torque_dset[i_step, 0] = self.torque1
            self._torque_dset[i_step, 1] = self.torque2
            self.written = True
        return i_step
