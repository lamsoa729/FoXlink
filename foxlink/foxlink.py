#!/usr/bin/env python
from matplotlib.animation import FFMpegWriter
from .animation_funcs import (makeAnimation, makeMinimalAnimation,
                              makeOrientAnimation, makeMomentAnimation,
                              makeMomentExpansionAnimation)
from .pde_analyzer import PDEAnalyzer
from .me_analyzer import MEAnalyzer
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
# Moment expansion solvers
from .me_solver import MomentExpansionSolver


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
                        help=("Print out more text from simulations. "
                              "NOT IMPLEMENTED YET!"))  # TODO
    parser.add_argument("-f", "--file", type=str, default="FP_params.yaml",
                        help=("Parameter file used to run FoXlink solver. "
                              "This should be a yaml parameter file if running a "
                              "simulation. If analyzing a file, this should be an "
                              "HDF5 file."))
    parser.add_argument("-t", "--test", action='store_true',
                        help="Run test protocol on FoXlink solver. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-c", "--change_params", action="store_true", default=False,
                        help="Change parameter file if options differ file values. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-a", "--analysis", type=str, default='',
                        help=("Perform post-processing analysis on simulations once completed."
                              "Necessary to make movies. Options: 'load', 'analyze', 'overwrite'.\n"
                              "\tload = Only add read data that has already been analyzed. \n"
                              "\tanalyze = Load data and analyze for values not previously analyzed. \n"
                              "\toverwrite = Re-analyze all values regardless of previous analysis \n"
                              "\tME = Analyze moment expansion \n")
                        )

    parser.add_argument("-g", "--graph", action="store_true", default=False,
                        help="Graph data after simulation has run and been analyzed. NOT IMPLEMENTED YET!")  # TODO
    parser.add_argument("-m", "--movie", type=str, default='',
                        help=("Make movie of systems of evolution. Options:\n"
                              "\tall = Make movie with all the data \n"
                              "\tmin = Make movie with just diagram of rods and crosslink data.\n"
                              "\torient = Make movie that describes orientation of rods.\n"
                              "\tmoment = Make movie that shows moments of crosslink distribution.\n"
                              "(Will try to load and analyze files if data is not there.)"))
    opts = parser.parse_args()
    return opts


class FoXlink(object):

    """!Control structure classs for FoXlink PDE solver framework. """

    def __init__(self, opts):
        """!Initialize FoXlink object with command line options.

        @param opts: ArgumentParser command line options

        """
        self._opts = opts

    def parse_params(self):
        """!Parse parameter file from options.

        @return: dictionary of parameters and options

        """
        with open(self._opts.file, 'r') as pfile:
            params = yaml.safe_load(pfile)

        return params

    def get_solver_type(self):
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

    def create_solver(self):
        """!Create solver based of solver type
        @return: solver object to run

        """
        try:
            solver = self._solver_type(pfile=self._opts.file)
        except BaseException:
            raise
        return solver

    def run(self):
        """!Run simulation with created solver
        @return: void

        """
        self._params = self.parse_params()
        self._solver_type = self.get_solver_type()
        self._solver = self.create_solver()
        self._solver.Run()
        self._solver.Save()

    def analyze(self):
        """!Analyze hdf5 file from foxlink solver run and make movie if parameter is given.
        @return: void

        """
        # analysis = FPAnalysis(self._opts.file)
        if self._opts.analysis:
            if self._opts.analysis == "ME":
                analyzer = MEAnalyzer(self._opts.file, 'overwrite')
            else:
                analyzer = PDEAnalyzer(self._opts.file, self._opts.analysis)
        else:
            analyzer = PDEAnalyzer(self._opts.file, 'analyze')

        if self._opts.movie:
            print("Started making movie")
            Writer = FFMpegWriter
            writer = Writer(
                fps=25, metadata=dict(
                    artist='Me'), bitrate=1800)
            if self._opts.analysis == "ME":
                makeMomentExpansionAnimation(analyzer, writer)
            elif self._opts.movie == 'all':
                makeAnimation(analyzer, writer)
            elif self._opts.movie == 'min':
                makeMinimalAnimation(analyzer, writer)
            elif self._opts.movie == 'orient':
                makeOrientAnimation(analyzer, writer)
            elif self._opts.movie == 'moment':
                makeMomentAnimation(analyzer, writer)
        analyzer.save()


def main():
    """!Main function of foxlink
    @return: void

    """
    opts = parse_args()
    foxlink = FoXlink(opts)
    if opts.analysis or opts.movie:
        foxlink.analyze()
    else:
        foxlink.run()


##########################################
if __name__ == "__main__":
    main()
