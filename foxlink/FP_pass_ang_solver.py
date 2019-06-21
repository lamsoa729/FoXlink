#!/usr/bin/env python
from .solver import Solver


"""@package docstring
File: FP_pass_ang_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Class that solves the distribution of crosslinks in an angular system.
"""


class FPPassiveAngSolver(Solver):

    """!Solve the Fokker-Planck equation for passive crosslinkers using the
    using the leap frog method with 4 point laplacian.
    """

    def __init__(self, pfile=None, name='FP_pass_ang'):
        """!Set parameters for PDE to be solved including boundary conditions.

        @param pfile: parameter file path

        """
        Solver.__init__(self, pfile, name)

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.
        @return: void
        """
        Solver.ParseParams(self)
        self.phio = self._params["phio"]  # Angle between MTs

    def calcSourceMatrix(self):
        """TODO: Docstring for makeSourceMatrix.
        @return: TODO

        """
        self.src_mat = make_ang_source_mat(self.s1, self.s2,
                                           self.R_pos,
                                           self._params['ko'],
                                           self._params['co'],
                                           self._params['ks'],
                                           self._params['ho'],
                                           self._params["beta"])

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """
        Solver.makeDataframe(self)
        self._phi_dset = self._mt_grp.create_dataset(
            'phi', shape=(self._nframes, 1))

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        i_step = Solver.Write(self)
        self._phi_dset[i_step] = self.phio


##########################################
if __name__ == "__main__":
    pdes = FPPassiveLFSolver(sys.argv[1])
    pdes.Run()
    pdes.Save('FP_pass_LF.pickle')
