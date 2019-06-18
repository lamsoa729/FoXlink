#!/usr/bin/env python
# In case of poor (Sh***y) commenting contact adam.lamson@colorado.edu
# Basic
import sys
import os
# Testing
import unittest
# import pdb
# import time, timeit
# import line_profiler
# Analysis
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# import yaml
# from math import *
# Speed
# from numba import jit
# Other importing
# sys.path.append(os.path.join(os.path.dirname(__file__), '[PATH]'))
from FP_helpers import *


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPHelpersPerpTestCase(unittest.TestCase):

    """!Docstring for FPHelpersTestCase. """

    def setUp(self):
        """!TODO: to be defined1. """
        # MT params
        r = 1.
        a1 = 0.
        a2 = 1.
        b = 0.
        # Crosslinker params
        ks = 1.
        ho = 1.
        self.f_args = [r, a1, a2, b, ks, ho]
        # Crosslinker velocity params
        self._vo = 1.
        self._fs = 1.

    def test_1_head_pos(self):
        """! Heads are positioned at s1 = 1, s2 = 0
        """
        fs = spring_force(1., 0., *self.f_args)
        self.assertAlmostEqual(fs, -0.4142135623730952)
        # self.assertEqual(fs, -0.4142135623730952)

    def test_2_head_pos(self):
        """! Heads are positioned at s1 = 0, s2 = -1, ie ontop of one another
        """
        fs = spring_force(0., -1., *self.f_args)
        self.assertAlmostEqual(fs, 1.0)

    def test_3_head_pos(self):
        """! Heads are positioned at s1 = 0, s2 = 0, ie in a line
        """
        fs = spring_force(0., 0, *self.f_args)
        self.assertAlmostEqual(fs, 0.0)


class FPHelpersAPTestCase(unittest.TestCase):

    """!Rods are antiparallel and separated perpendicular to there axes by 1. """

    def setUp(self):
        # MT params
        r = 1.
        a1 = 0.
        a2 = 0.
        b = -1.
        # Crosslinker params
        ks = 1.
        ho = 1.
        self._vo = 1.
        self._fs = 1.
        # Force arguments
        self.f_args = [r, a1, a2, b, ks, ho]

    def test_1_head_pos(self):
        """!
        """
        self.assertEqual(1, 1)

    # def test_2_head_pos(self):
    #     """!
    #     """
    #     pass

    # def test_3_head_pos(self):
    #     """!
    #     """
    #     pass


class FPHelpers45DegTestCase(unittest.TestCase):

    """!Rods are antiparallel and separated perpendicular to there axes by 1. """

    def setUp(self):
        # MT params
        self._r = 1.
        self._a1 = 0.
        self._a2 = 0.
        self._b = -1.
        # Crosslinker params
        self._ks = 1.
        self._ho = 1.
        self._vo = 1.
        self._fs = 1.

    def test_1_head_pos(self):
        """!
        """
        self.assertEqual(1, 1)

    # def test_2_head_pos(self):
    #     """!
    #     """
    #     pass

    # def test_3_head_pos(self):
    #     """!
    #     """
    #     pass


##########################################
if __name__ == "__main__":
    suite_perp = unittest.TestLoader().loadTestsFromTestCase(FPHelpersPerpTestCase)
    suite_ap = unittest.TestLoader().loadTestsFromTestCase(FPHelpersAPTestCase)
    suite_45deg = unittest.TestLoader().loadTestsFromTestCase(FPHelpers45DegTestCase)
    alltests = unittest.TestSuite([suite_perp, suite_ap, suite_45deg])
    unittest.TextTestRunner(verbosity=2).run(alltests)
