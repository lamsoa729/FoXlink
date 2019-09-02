#!/usr/bin/env python
from matplotlib.animation import FFMpegWriter
from .animation_funcs import (makeAnimation, makeMinimalAnimation,
                              makeOrientAnimation)
from .FP_analysis import FPAnalysis
# from .FP_pass_ang_CN import FPPassiveAngCNSolver
import argparse
import sys
import yaml

# Import all solvers
# Orient
from .FP_gen_orient_static_xlinks_solver import FPGenOrientStaticXlinksSolver
from .FP_gen_orient_motor_UW_solver import FPGenOrientMotorUWSolver
# Free motion solvers
from .FP_gen_motion_static_xlinks_solver import FPGenMotionStaticXlinksSolver
from .FP_gen_motion_pass_CN_solver import FPGenMotionPassCNSolver
from .FP_gen_motion_motor_UW_solver import FPGenMotionMotorUWSolver
# Optical trap solvers
from .FP_OT_gen_motion_motor_UW_solver import FPOpticalTrapGenMotionMotorUWSolver
from .FP_OT_gen_motion_static_xlinks_solver import FPOpticalTrapGenMotionStaticXlinksSolver
# Defined motion
from .FP_gen_def_motion_motor_UW_solver import FPGenDefMotionMotorUWSolver


"""@package docstring
File: foxlink.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Main control program for FoXlink PDE solver. Parses arguments using
argsparse. Type foxlink -h for help and main actions.
"""


def parse_args():
    parser = argparse.ArgumentParser(prog='foxlink.py',
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Specialized optical trap arguments go here
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Print out more text from simulations. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-f", "--file", type=str, default="FP_params.yaml",
                        help="Parameter file used to run FoXlink solver. This should be a yaml parameter file if running a simulation. If analyzing a file, this should be an HDF5 file.")
    parser.add_argument("-t", "--test", action='store_true',
                        help="Run test protocol on FoXlink solver. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-c", "--change_params", action="store_true", default=False,
                        help="Change parameter file if options differ file values. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-a", "--analyze", action="store_true", default=False,
                        help="Perform post-analysis on simulations once completed. Necessary to make movies.")
    parser.add_argument("-g", "--graph", action="store_true", default=False,
                        help="Graph data after simulation has run and been analyzed. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-m", "--movie", type=str, default='',
                        help=("Make movie of systems of evolution. Options: 'all' or 'min'.\n"
                              "\tall = Make movie with all the data \n"
                              "\tmin = Make movie with just diagram of rods and crosslink data.\n"
                              "(Will analyze files again.)"))
    opts = parser.parse_args()
    return opts


class FoXlink(object):

    """!Control structure classs for FoXlink PDE solver framework. """

    def __init__(self, opts):
        """!Initialize FoXlink object with command line options.

        @param opts: ArgumentParser command line options

        """
        self._opts = opts

    def ParseParams(self):
        """!Parse parameter file from options.

        @return: dictionary of parameters and options

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
            solver = self._solver_type(pfile=self._opts.file)
        except BaseException:
            raise
        return solver

    def Run(self):
        """!Run simulation with created solver
        @return: void

        """
        self._params = self.ParseParams()
        self._solver_type = self.getSolverType()
        self._solver = self.createSolver()
        self._solver.Run()
        self._solver.Save()

    def Analyze(self):
        """!Analyze hdf5 file from foxlink solver run and make movie if parameter is given.
        @return: void

        """
        analysis = FPAnalysis(self._opts.file)
        if self._opts.analyze:
            analysis.Analyze(True)
        if self._opts.movie:
            print("Started making movie")
            Writer = FFMpegWriter
            writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
            if self._opts.movie == 'all':
                makeAnimation(analysis, writer)
            elif self._opts.movie == 'min':
                makeMinimalAnimation(analysis, writer)
            elif self._opts.movie == 'orient':
                makeOrientAnimation(analysis, writer)

        analysis.Save()


def main():
    """!Main function of foxlink
    @return: void

    """
    opts = parse_args()
    FXlink = FoXlink(opts)
    if opts.analyze or opts.movie:
        FXlink.Analyze()
    else:
        FXlink.Run()


##########################################
if __name__ == "__main__":
    main()
