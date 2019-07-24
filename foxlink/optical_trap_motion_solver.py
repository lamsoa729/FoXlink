#!/usr/bin/env python
from .rod_motion_solver import RodMotionSolver

"""@package docstring
File: optical_trap_motion_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
import yaml
import h5py


class OpticalTrapMotionSolver(RodMotionSolver):

    """!Docstring for FPGenOpticalTrapMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: parameter file
        @param pdict: parameter dictionary

        """
        print("Init OpticalTrapMotionSolver ->", end=" ")
        FPRodMotionSolver.__init__(self, pfile=pfile, pdict=pdict)
        self.OTParseParams()

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

    def RodStep(self, force1=0, force2=0, torque1=0, torque2=0,
                R1_pos=None, R2_pos=None, R1_vec=None, R2_vec=None):
        """! Change the position of rods based on forces and torques exerted on rod
        @param force: Force vector of rod2 by rod1
        @param torque: Torque vector of rod2 by rod1
        @param R1_pos: TODO
        @param R2_pos: TODO
        @param R1_vec: TODO
        @param R2_vec: TODO
        @return: void
        @return: TODO

        """
        self.calcOTInteractions(R1_pos, R2_pos, R1_vec, R2_vec)
        return FPRodMotionSolver.RodStep(self,
                                         force1 + self.ot1_force,
                                         force2 + self.ot2_force,
                                         torque1 + self.ot1_torque,
                                         torque2 + self.ot2_torque,
                                         R1_pos, R2_pos, R1_vec, R2_vec)

    def calcOTInteractions(self, R1_pos, R2_pos, R1_vec, R2_vec):
        """!TODO: Docstring for calcOTInteractions.

        @param R1_pos: TODO
        @param R2_pos: TODO
        @param R1_vec: TODO
        @param R2_vec: TODO
        @return: TODO

        """
        ot_k = self._params['OT_ks']
        hL1 = .5 * self._params["L2"]
        hL2 = .5 * self._params["L2"]
        # Get position of minus ends
        rod1_minus_pos = R1_pos - (hL1 * R1_vec)
        rod2_minus_pos = R2_pos - (hL2 * R2_vec)

        self.ot1_force = -ot_k * (rod1_minus_pos - self.OT1_pos)
        self.ot2_force = -ot_k * (rod2_minus_pos - self.OT2_pos)
        self.ot1_torque = np.cross(-hL1 * R1_vec, self.ot1_force)
        self.ot2_torque = np.cross(-hL2 * R2_vec, self.ot2_force)

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

        self._ot_grp = self._h5_data.create_group('OT_data')
        self._ot_grp.create_dataset('OT1_pos', shape=(self._nframes + 1, 3))
        self._ot_grp.create_dataset('OT2_pos', shape=(self._nframes + 1, 3))

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        i_step = FPRodMotionSolver.Write(self)
        self.OTWrite(i_step)
        return i_step

    def OTWrite(self, i_step):
        self._ot_force_dset[i_step, 0] = self.ot1_force
        self._ot_force_dset[i_step, 1] = self.ot2_force
        self._ot_torque_dset[i_step, 0] = self.ot1_torque
        self._ot_torque_dset[i_step, 1] = self.ot2_torque
