#!/usr/bin/env python
from copy import deepcopy as dcp
from math import sin
import numpy as np

"""@package docstring
File: optical_trap_types.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class OpticalTrapType(object):

    """!Specify behaviour of optical trap object"""

    def __init__(self, params, ot_num):
        """!Initialize parameters responsible for optical trap motion

        @param params: parameters for optical trap motion

        """
        self._params = params
        self._ot_num = ot_num

    def moveOpticalTrap(self, slvr, pos):
        """!Calculate the new position of the optical trap

        @param slvr: Solver object where most parameters will come from
        @param pos: current position of optical trap
        @param ot_num: index of the optical trap (optional)
        @return: the new position of the optical trap

        """
        print("!!!Warning: moveOpticalTrap has not been implemented for "
              "{} class.".format(self.__class__.__name__))
        new_pos = pos
        return new_pos


class OpticalTrapOscillator(OpticalTrapType):

    """!Trap that oscillates around a fix positioned"""

    def __init__(self, pos, params, ot_num):
        """!Initialize oscillating trap with direction, frequency, phase,
        and amplitude.
        """
        OpticalTrapType.__init__(self, params, ot_num)
        self.init_pos = np.asarray(pos)
        self.osc_ax = np.asarray(params['direction'])
        self.freq = params['frequency']
        self.phase = params['phase']
        self.amp = params['amplitude']

    def moveOpticalTrap(self, slvr, pos):
        """!Move trap to position based on time of simulation

        @param slvr: TODO
        @param pos: TODO
        @return: TODO

        """
        return self.init_pos + (self.osc_ax * self.amp *
                                sin(self.freq * slvr.t - self.phase))
