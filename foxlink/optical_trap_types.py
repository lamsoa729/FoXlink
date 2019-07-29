#!/usr/bin/env python

"""@package docstring
File: optical_trap_types.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class OpticalTrapType(object):

    """!Specify behaviour of optical trap object"""

    def __init__(self, params):
        """!Initialize parameters responsible for optical trap motion

        @param params: TODO

        """
        self._params = params

    def moveOpticalTrap(self, slvr, pos, ot_num=None):
        """!Calculate the new position of the optical trap

        @param slvr: Solver object where most parameters will come from
        @param pos: current position of optical trap
        @param ot_num: index of the optical trap (optional)
        @return: the new position of the optical trap

        """
        print("!!!Warning: moveOpticalTrap has not been implemented for "
              "{} class.".format(self.__class__.__name__))
        return new_pos
