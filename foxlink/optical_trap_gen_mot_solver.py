#!/usr/bin/env python
# from .FP_gen_motion_solver import FPGenMotionSolver
from .FP_gen_orient_solver import FPGenOrientSolver

"""@package docstring
File: FP_optical_trap_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
import yaml
import h5py


class OpticalTrapGenOrientSolver(FPGenOrientSolver):

    """!Docstring for FPGenOpticalTrapMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: parameter file
        @param pdict: parameter dictionary

        """
        print("Init FPGenOpticalTrapMotionSolver ->", end=" ")
        self.ParseParams(skip=True)
        self.OTParseParams()
        FPGenOrientSolver.__init__(self, pfile=pfile, pdict=pdict)

    def OTParseParams(self):
        """!TODO: Docstring for ParseParams.

        @return: void, set position of optical traps and motion

        """
        if 'OT1_pos' in self._params:  # Set to defined location
            self.OT1_pos = np.asarray(self._params['OT1_pos'])
        else:  # Set optical trap 2 to minus end of rod1
            hL1 = .5 * self._params["L2"]
            self.OT1_pos = self.R1_pos - (hL1 * self.R1_vec)

        if 'OT2_pos' in self._params:  # Set to defined location
            self.OT2_pos = np.asarray(self._params['OT2_pos'])
        else:  # Set optical trap 2 to minus end of rod2
            hL2 = .5 * self._params["L2"]
            self.OT2_pos = self.R2_pos - (hL2 * self.R2_vec)

        if 'OT_ks' not in self._params:
            raise KeyError('OT_k must be defined for optical trap runs')

        # TODO add motion to optical traps
        # if 'OT1_motion' in self._params:
        # if 'OT2_motion' in self._params:

    def calcForceMatrix(self):
        """!Calculate the force for each crosslinker
        @return: TODO

        """
        FPGenOrientSolver.calcForceMatrix(self)
        # Add in force from optical traps

        # Parameters for calculations
        ot_k = self._params['OT_ks']
        hL1 = .5 * self._params["L2"]
        hL2 = .5 * self._params["L2"]
        # Get position of minus ends
        rod1_minus_pos = self.R1_pos - (hL1 * self.R1_vec)
        rod2_minus_pos = self.R2_pos - (hL2 * self.R2_vec)

        self.ot1_force = -ot_k * (rod1_minus_pos - self.OT1_pos)
        self.ot2_force = -ot_k * (rod2_minus_pos - self.OT2_pos)

    def calcTorqueMatrix(self):
        """! Calculate the torque provided by each point
        @return: TODO

        """
        FPGenOrientSolver.calcTorqueMatrix(self)
        # Parameters for calculations
        ot_k = self._params['OT_ks']
        hL1 = .5 * self._params["L2"]
        hL2 = .5 * self._params["L2"]
        # Calculate torque
        self.ot1_torque = np.cross(-hL1 * self.R1_vec, self.ot1_force)
        self.ot2_torque = np.cross(-hL2 * self.R2_vec, self.ot2_force)

    def stepOT(self):
        """!TODO: Docstring for stepOT.
        @return: TODO

        """
        print("stepOT still needs to be implemented")
        pass

    def addOTDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """
        if not self.data_frame_made:
            FPGenOrientSolver.makeDataframe(self)
        self._ot_force_dset = self._interaction_grp.create_dataset(
            'optical_trap_force_data',
            shape=(self._nframes + 1, 2, 3),
            dtype=np.float32)
        for dim, label in zip(self._ot_force_dset.dims,
                              ['frame', 'trap', 'coord']):
            dim.label = label
        self._ot_torque_dset = self._interaction_grp.create_dataset(
            'optical_trap_torque_data',
            shape=(self._nframes + 1, 2, 3),
            dtype=np.float32)
        for dim, label in zip(self._ot_torque_dset.dims,
                              ['frame', 'trap', 'coord']):
            dim.label = label

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        if not self.written:
            i_step = FPGenOrientSolver.Write(self)
        else:
            i_step = ((self.t / self.dt) / self.nwrite)
        self._ot_force_dset[i_step, 0] = self.ot1_force
        self._ot_force_dset[i_step, 1] = self.ot2_force
        self._ot_torque_dset[i_step, 0] = self.ot1_torque
        self._ot_torque_dset[i_step, 1] = self.ot2_torque
        return i_step
