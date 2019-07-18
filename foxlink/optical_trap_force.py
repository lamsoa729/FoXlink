#!/usr/bin/env python

"""@package docstring
File: FP_optical_trap_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
import yaml
import h5py


class FPGenOpticalTrapMotionSolver(FPGenMotionSolver):

    """!Docstring for FPGenOpticalTrapMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: parameter file
        @param pdict: parameter dictionary

        """
        print("Init FPGenOpticalTrapMotionSolver ->", end=" ")
        FPGenMotionSolver.__init__(self, pfile=pfile, pdict=pdict)

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.

        @return: void, set position of optical traps and motion

        """
        FPGenOrientSolver.ParseParams(self)
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
        FPGenMotionSolver.calcForceMatrix()
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
        FPGenMotionSolver.calcTorqueMatrix()
        # Parameters for calculations
        ot_k = self._params['OT_ks']
        hL1 = .5 * self._params["L2"]
        hL2 = .5 * self._params["L2"]
        # Calculate torque
        self.ot1_torque = np.cross(-hL1 * self.R1_vec, self.ot1_force)
        self.ot2_torque = np.cross(-hL2 * self.R2_vec, self.ot2_force)

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
