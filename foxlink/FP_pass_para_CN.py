#!/usr/bin/env python
# Other importing
from .FP_pass_para_solver import FPPassiveParaSolver
from .FP_pass_CN_solver import FPPassiveCNSolver
from .FP_initial_conditions import *


"""@package docstring
File:
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


class FPPassiveParaCNSolver(FPPassiveCNSolver, FPPassiveParaSolver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the Crank-Nicholson method with 4 point laplacian.
    """

    def __init__(self, pfile=None, name="FP_pass_para_CN"):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path
        @param name: name to store data under

        """
        FPPassiveCNSolver.__init__(self, pfile, name)


##########################################
if __name__ == "__main__":
    pdes = FPPassiveParaCNSolver(sys.argv[1])
    t0 = time.time()
    pdes.Run()
    t1 = time.time()
    print(" Run took {} for {} steps.".format(t1 - t0, pdes.nsteps))
    pdes.Save()
    # pdes.Save('FP_pass_LF.pickle')
