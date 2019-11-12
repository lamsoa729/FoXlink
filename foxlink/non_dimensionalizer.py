#!/usr/bin/env python

"""@package docstring
File: non_dimensionalizer.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""

import numpy as np
from copy import deepcopy as dcp


class NonDimensionalizer():

    """!Class to non-dimensionalize parameters."""

    def __init__(self, **kwargs):
        """!Take in the necessary values to non-dimensionalize any quantitiy.

        @param kwargs

        """
        self.__dict__.update(kwargs)

    def non_dim_val(self, value, dim_list, exp_list=None):
        """!Non-dimensionalize value based on dimensions and dimension powers.

        @param value: Value you wish to convert do a dimensionless quantity. Maybe a numpy array.
        @param dim_list: Dimensions. Must be in __dict__ of class
        @param exp_list: Exponent of dimensions given.
        @return: non-dimensionalized value

        """
        if exp_list is None:
            exp_list = [1]
        val = dcp(value)
        for dim, exp in zip(dim_list, exp_list):
            if dim not in self.__dict__:
                raise RuntimeError(
                    "Dimension ({}) not in NonDimensionalizer.".format(dim))
            val /= np.power(self.__dict__[dim], exp)
        print(val)
        return val

    def dim_val(self, value, dim_list, exp_list=None):
        """!Non-dimensionalize value based on units and dim

        @param value: Value you wish to convert from dimensionless quantity to
                      a dimensionful quanity. Maybe a numpy array.
        @param dim_list: Dimensions. Must be in __dict__ of class
        @param exp_list: Exponent of dimensions given.
        @return: TODO

        """
        if exp_list is None:
            exp_list = [1]
        val = dcp(value)
        for dim, exp in zip(dim_list, exp_list):
            if dim not in self.__dict__:
                raise RuntimeError(
                    "Dimension ({}) not in NonDimensionalizer.".format(dim))
            val *= np.power(self.__dict__[dim], exp)
        return val

    def calc_new_dim(self, dim_name, dim_list, exp_list):
        """!Calculate new dimension based off current dimensions that already
        exist and store it as a property.

        @param dim_name: TODO
        @param dim_list: TODO
        @param exp_list: TODO
        @return: void

        """
        val = 1.
        for dim, exp in zip(dim_list, exp_list):
            if dim not in self.__dict__:
                raise RuntimeError(
                    "Dimension ({}) not in NonDimensionalizer.".format(dim))
            val *= np.power(self.__dict__[dim], exp)
        self.__dict__[dim_name] = val
