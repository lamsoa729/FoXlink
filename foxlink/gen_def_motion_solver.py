#!/usr/bin/env python
from .rod_motion_solver import RodMotionSolver
import numpy as np
"""@package docstring
File: gen_def_rod_motion.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class DefinedMotion(object):

    """!Docstring for DefinedMotion. """

    def __init__(self, mot_params):
        """!Returns positions given a time

        @param mot_params: Define position of rods as a function of time.

        """
        self._mot_params = mot_params
        self.pos_funcs = None
        self.vec_funcs = None
        if 'pos' in self._mot_params:
            self.pos_funcs = self._mot_params['pos']
        if 'vec' in self._mot_params:
            self.vec_funcs = self._mot_params['vec']

    def getPos(self, t):
        """! Return positions based on functions

        @param t: Current time
        @return: Current position

        """
        return np.asarray([eval(str(p), {'t': t}) for p in self.pos_funcs])

    def getVec(self, t):
        """!Gets orientation of rods based on functions

        @param t: Current time
        @return: Current orientation

        """
        return np.asarray([eval(str(v), {'t': t}) for v in self.vec_funcs])


class GenDefMotionSolver(RodMotionSolver):

    """!Docstring for GenDefMotionSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!TODO: to be defined1.

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init GenDefMotionSolver ->", end=" ")
        RodMotionSolver.__init__(self)
        self.DefMotionParseParams()

    def DefMotionParseParams(self):
        """!Parse parameters specific to definite motion class
        @return: TODO

        """
        # self.R1_mot = None
        # self.R2_mot = None
        if 'R1_mot' in self._params:
            self.R1_mot = DefinedMotion(self._params['R1_mot'])
        else:
            self.R1_mot = DefinedMotion({})

        if 'R2_pos_mot' in self._params:
            self.R2_mot = DefinedMotion(self._params['R2_mot'])
        else:
            self.R2_mot = DefinedMotion({})

    def RodStep(self, *args, **kwargs):
        """! Change the position of rods based on forces and torques exerted on rod
        @return: void
        @return: TODO

        """
        # Change center of mass positions
        if self.R1_mot.pos_funcs is not None:
            self.R1_pos = self.R1_mot.getPos(self.t)
        if self.R2_mot.pos_funcs is not None:
            self.R2_pos = self.R2_mot.getPos(self.t)
        # Change orientation vectors
        if self.R1_mot.vec_funcs is not None:
            self.R1_vec = self.R1_mot.getVec(self.t)
        if self.R2_mot.vec_funcs is not None:
            self.R2_vec = self.R2_mot.getVec(self.t)

        return self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec
