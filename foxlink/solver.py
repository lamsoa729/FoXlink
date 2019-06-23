#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
# from os import path
# Testing
from pathlib import Path
import time
import numpy as np
import yaml
import h5py
from scipy import sparse


"""@package docstring
File: solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Base Solver class for FoXlink
"""


class Solver(object):

    """!Docstring for Solver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions

        @param pfile: parameter file for PDEs

        """
        print("Init Solver")
        self._pfile = pfile
        self._params = pdict
        self.data_frame_made = False
        self.written = False
        # Initialize parameters from file
        self.ParseParams()
        self.makeSolutionGrid()
        self.setInitialCondition()
        # Create data frame
        self._h5_data = h5py.File(
            Path("{}.h5".format(self.__class__.__name__)), 'w')
        # Add meta data to hdf5 file
        for k, v in self._params.items():
            self._h5_data.attrs[k] = v
        self.makeDataframe()

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.

        @return: TODO

        """
        if self._pfile is not None:
            with open(self._pfile, 'r') as pf:
                self._params = yaml.safe_load(pf)
        elif self_params is None:
            self._params = default_params

        # Integration parameters
        self.t = 0.
        self.ds = self._params["ds"]  # Segmentation size of microtubules
        self.nwrite = self._params["nwrite"]
        if "nt" not in self._params:
            self.nsteps = self._params["nsteps"]
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
            self.nsteps = float(self.nt / self.dt)
            self._params["nsteps"] = self.nsteps
        # Make time array. Set extra space for initial condition
        self.time = np.linspace(0, self.nt, self.nsteps + 1).tolist()
        print("Time step: ", self.dt)
        print("Total time: ", self.nt)
        print("Number of steps: ", self.nsteps)

    def setInitialCondition(self):
        """!TODO: Docstring for setInitialCondition.
        @return: TODO

        """
        if 'initial_condition' in self._params:
            if self._params['initial_condition'] == 'equil':
                self.sgrid += self.src_mat / self._params['ko']
            elif self._params['initial_condition'] == 'empty':
                pass
            else:
                self.sgrid += eval(self._params['initial_condition'])
            print(self.sgrid)

    def makeSolutionGrid(self):
        """!Make an array of solutions to solve the PDE
        @return: void

        """
        L1 = self._params["L1"]
        L2 = self._params["L2"]

        ds = self.ds
        self.ns1 = int(L1 / ds) + 2
        self.ns2 = int(L2 / ds) + 2

        # TODO This maintains proper spacing but I should check this with
        #      someone who understands methods better
        self._params["L1"] = ds * (self.ns1 - 2)
        self._params["L2"] = ds * (self.ns2 - 2)

        # Discrete rod locations, extra spots left for boundary conditions
        self.s1, step1 = np.linspace(
            0, ds * (self.ns1 - 1), self.ns1, retstep=True)
        self.s1 -= L1 * .5
        self.s2, step2 = np.linspace(
            0, ds * (self.ns2 - 1), self.ns2, retstep=True)
        self.s2 -= (L2 * .5)
        print("ds1: ", step1)
        print("ds2: ", step2)

        # self.sgrid = sparse.csc_matrix((self.ns1, self.ns2))
        self.sgrid = np.zeros((self.ns1, self.ns2))
        self.calcSourceMatrix()
        self.calcForceMatrix()
        self.calcTorqueMatrix()

    def Run(self):
        """!Run PDE solver with parameters in pfile
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
            if (int(self.t / self.dt) % self.nwrite) == 0:
                t1 = time.time()
                print(r" {} steps in {:.4f} seconds, {:.1f}% complete".format(
                    self.nwrite, t1 - t0, float(self.t / self.nt) * 100.))
                self.Write()
                # Reset write function
                self.written = False
                t0 = time.time()
        print(r" --- Total of {} steps in {:.4f} seconds ---".format(self._nsteps, 
                    time.time()-t_start))
        return

    def Step(self):
        """!Step solver method one unit in time
        @return: Sum of changes of grid, Changes phi1 and phi0

        """
        print("Step not made!")

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: Dictionary pointing to data in dataframe

        """
        # Enter params into hdf5 data file as attributes for later
        if not self.data_frame_made:
            for key, param in self._params.items():
                self._h5_data.attrs[key] = param
            time = self.time[::self.nwrite]
            self._nframes = len(time)
            self._time_dset = self._h5_data.create_dataset('time', data=time,
                                                           dtype=np.float32)
            self._xl_grp = self._h5_data.create_group('XL_data')
            self._mt_grp = self._h5_data.create_group('MT_data')

            self._mt_grp.create_dataset('s1', data=self.s1)
            self._mt_grp.create_dataset('s2', data=self.s2)

            self._xl_distr_dset = self._xl_grp.create_dataset(
                'XL_distr',
                shape=(self.ns1, self.ns2, self._nframes + 1),
                dtype=np.float32)

            self._interaction_grp = self._h5_data.create_group(
                'Interaction_data')
            self._force_dset = self._interaction_grp.create_dataset(
                'force_data',
                shape=(self._nframes + 1, 3),
                dtype=np.float32)
            self._torque_dset = self._interaction_grp.create_dataset(
                'torque_data',
                shape=(self._nframes + 1, 3),
                dtype=np.float32)
            self.data_frame_made = True

    def calcSourceMatrix(self):
        """Virtual functions for calculating source matrix
        @return: TODO

        """
        print("calcSourceMatrix not implemented. Source matrix initialized with zeros.")
        self.src_mat = sparse.csc_matrix((self.ns1, self.ns2))

    def calcForceMatrix(self):
        """Virtual functions for calculating force matrix for a given configuration
        @return: TODO

        """
        print("calcforceMatrix not implemented. Source matrix initialized with zeros.")
        self.f_mat = sparse.csc_matrix((self.ns1, self.ns2, 3))

    def calcTorqueMatrix(self):
        """Virtual functions for calculating force matrix for a given configuration
        @return: TODO

        """
        print("calcTorqueMatrix not implemented. Source matrix initialized with zeros.")
        self.t_mat = sparse.csc_matrix((self.ns1, self.ns2, 3))

    def Write(self):
        """!Write current step in algorithm into data frame
        @return: void

        """
        i_step = ((self.t / self.dt) / self.nwrite)
        if not self.written:
            # self._xl_distr_dset[:, :, i_step] = self.sgrid.todense()
            self._xl_distr_dset[:, :, i_step] = self.sgrid
            self._time_dset[i_step] = self.t
            self._force_dset[i_step] = self.force
            self._torque_dset[i_step] = self.torque
            self.written = True
        return i_step

    def Save(self):
        """!Flush and close hdf5 data frame
        @return: void

        """
        self._h5_data.flush()
        self._h5_data.close()


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
