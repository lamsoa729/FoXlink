#!/usr/bin/env python
import argparse
import sys
import yaml

# Import all solvers
from .FP_gen_motion_static_xlinks import FPGenMotionStaticXlinks
from .FP_static_solver import FPStaticSolver
from .FP_pass_para_CN import FPPassiveParaCNSolver
from .FP_pass_ang_CN import FPPassiveAngCNSolver


"""@package docstring
File: foxlink.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def parse_args():
    parser = argparse.ArgumentParser(prog='foxlink.py')
    # Specialized optical trap arguments go here
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Print out more text from simulations.")  # TODO
    parser.add_argument("-f", "--file", type=str, default="params.yaml",
                        help="Parameter file used to run FoXlink solver.")
    parser.add_argument("-t", "--test", action='store_true',
                        help="Run test protocol on FoXlink solver.")  # TODO
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Specify output file name for generated data.")  # TODO
    parser.add_argument("-c", "--change_params", action="store_true", default=False,
                        help="Change parameter file if options differ file values.")  # TODO
    parser.add_argument("-a", "--analyze", action="store_true", default=False,
                        help="Perform post-analysis on simulations once completed.")  # TODO
    parser.add_argument("-g", "--graph", action="store_true", default=False,
                        help="Graph data after simulation has run and been analyzed.")  # TODO
    opts = parser.parse_args()
    return opts


class FoXlink(object):

    """!Control structure for FoXlink solver. """

    def __init__(self, opts):
        """!Initialize FoXlink object with options

        @param opts: ArgumentParser command line options

        """
        self._opts = opts
        self._params = self.ParseParams()
        self._solver_type = self.getSolverType()
        self._solver = self.createSolver()

    def ParseParams(self):
        """!Parse parameter file from options.

        @return: dictionary of options

        """
        with open(self._opts.file, 'r') as pf:
            params = yaml.safe_load(pf)

        return params

    def getSolverType(self):
        """!Figure out which solver type to use on parameters
        @return: Class of a solver type

        """
        try:
            solver_type = getattr(sys.modules[__name__],
                                  self._params['solver_type'])
        except KeyError:
            raise
        except ImportError:
            raise
        return solver_type

    def createSolver(self):
        """!Create solver based of solver type
        @return: solver object to run

        """
        try:
            solver = self._solver_type(self._opts.file, name=self._opts.output)
        except BaseException:
            raise
        return solver

    def Run(self):
        """!Run simulation with created solver
        @return: void

        """
        self._solver.Run()
        self._solver.Save()


def main():
    """!Main function of foxlink
    @return: void

    """
    opts = parse_args()
    FXlink = FoXlink(opts)
    FXlink.Run()


##########################################
if __name__ == "__main__":
    main()
