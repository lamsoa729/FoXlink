#!/usr/bin/env python
import sys
from copy import deepcopy as dcp
import numpy as np
from scipy.integrate import dblquad
import yaml
from .FP_helpers import *


"""@package docstring
File: FP_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Solve Fokker-Planck equation
"""

default_params = {
    "r": 1,  # Distance between rod centers
    "a1": 0,  # Dot product between u1 and r unit vectors
    "a2": 0,  # Dot product between u2 and r unit vectors
    "b": -1,  # Dot product between u1 and u2 unit vectors
    "L1": 100,  # Length of microtubule 1
    "L2": 100,  # Length of microtubule 2
    "dt": 1,  # Time step
    "nt": 2000,  # total time
    "ds": 1,  # Segmentation size of microtubules
    "ko": 1,  # Crosslinker turnover rate
    "co": 1,  # Effective crosslinker concentration
    "ks": 1,  # Crosslinker spring concentration
    "ho": 1,  # Equilibrium length of crosslinkers
    "vo": 1,  # Base velocity of crosslinker heads
    "fs": 1,  # Stall force of crosslinker heads
    "beta": 1,  # Inverse temperature
}


class PDESolver(object):

    """!Docstring for PDESolver. """

    def __init__(self, pfile=None):
        """!Set parameters for PDE to be solved including boundary conditions

        @param pfile: parameter file for PDEs

        """
        self._pfile = pfile
        self._params = None
        self.t = 0  # current time
        self.ParseParams()

    def ParseParams(self):
        """!TODO: Docstring for ParseParams.

        @return: TODO

        """
        if self._pfile is not None:
            with open(self._pfile, 'r') as pf:
                self._params = yaml.safe_load(pf)
        else:
            self._params = default_params
        self.L1 = self._params["L1"]  # Length of microtubule 1
        self.L2 = self._params["L2"]  # Length of microtubule 2

        # Integration parameters
        self.dt = self._params["dt"]  # Time step
        self.nt = self._params["nt"]  # total time
        self.ds = self._params["ds"]  # Segmentation size of microtubules
        # Boundary conditions

        # Random other options
        print("Parsing params")

        self.MakeSolutionGrid()

        if self._params["type"] == "stationary":
            self.stationary_flag = True
            self.LUT, self.sparseT = self.MakeBoltzLUT()
            print(self.LUT)
        else:
            self.stationary_flag = False
            self.LUT = None
            self.sparseT = None

    def MakeSolutionGrid(self):
        """!TODO: Docstring for MakeMTGrid.
        @return: TODO

        """
        L1 = self.L1
        L2 = self.L2
        ds = self.ds

        self.ns1, self.ns2 = (int(L1 / ds), int(L2 / ds))
        # Discrete rod locations
        self.s1 = np.linspace(-.5 * L1, .5 * L1 - ds, self.ns1) + (ds * .5)
        self.s2 = np.linspace(-.5 * L2, .5 * L2 - ds, self.ns2) + (ds * .5)

        # Solution grids
        # Current solution time step
        self.phi0 = np.zeros((self.ns1, self.ns2))
        self.phi1 = np.zeros((self.ns1, self.ns2))  # Next solution time step

    def MakeBoltzLUT(self):
        """! Make lookup table for boltzmann factors for cell indices.
        @return: 2D array of integrated boltzmann factors

        """
        LUT = np.zeros((self.ns1, self.ns2))
        sparseT = np.zeros((self.ns1, self.ns2))
        ds_h = .5 * self.ds

        b_args = [
            self._params["r"],
            self._params["a1"],
            self._params["a2"],
            self._params["b"],
            self._params["ks"],
            self._params["ho"],
            self._params["beta"],
        ]
        # If boltzmann factor is smaller than 10e-6 and force is larger than
        # 5 times the stall force, skip calculation
        fc = sqrt(12. * self._params["ks"] * log(10.) / self._params["beta"])
        for i in range(self.ns1):
            for j in range(self.ns2):
                s1 = self.s1[i]
                s2 = self.s2[j]
                force = spring_force(s1, s1, *(b_args[:-1]))
                if abs(force) > fc and force > (-5. * self._params["fs"]):
                    LUT[i, j] = 0.0
                    sparseT[i, j] = 0
                else:
                    LUT[i, j], error = dblquad(boltz_fact, s1 - ds_h, s1 + ds_h,
                                               lambda r2: s2 - ds_h, lambda r2: s2 + ds_h,
                                               args=b_args)
                    # args=b_args, epsabs=0, epsrel=1e-8)
                    sparseT[i, j] = 1
                    print(
                        ("{}, {} = {} p/m {}".format(i, j, LUT[i, j], error)))

        return LUT, sparseT

    def Run(self):
        """!Run PDE solver with parameters in pfile
        @return: void

        """
        while self.t < self.nt:
            self.Step()
            self.t += self.dt
            if ((self.t / self.dt) % self._params['n_write']) == 0:
                self.Write()
            # print self.t

        return

    def Step(self):
        """!Use forward Eurler method to step forward one instance in dt
        @return: Sum of changes of grid, Changes phi1 and phi0

        """
        # Easier coefficients to work with
        ds_h = self.ds * .5
        dt = self.dt
        CFL = dt / self.ds
        s1 = self.s1
        s2 = self.s2

        r = self._params["r"]  # distance between rod centers
        a1 = self._params["a1"]  # dot product between u1 and r unit vectors
        a2 = self._params["a2"]  # dot product between u2 and r unit vectors
        b = self._params["b"]  # dot product between u1 and u2 unit vectors
        ko = self._params["ko"]  # Crosslinker turnover rate
        co = self._params["co"]  # Effective crosslinker concentration
        ks = self._params["ks"]  # Crosslinker spring concentration
        ho = self._params["ho"]  # Equilibrium length of crosslinkers
        vo = self._params["vo"]  # Base velocity of crosslinker heads
        fs = self._params["fs"]  # Stall force of crosslinker heads
        beta = self._params["beta"]  # Inverse temperature
        f_args = [r, a1, a2, b, ks, ho]
        b_args = [r, a1, a2, b, ks, ho, beta]

        for i in range(1, self.ns1):
            for j in range(1, self.ns2):
                #  TODO: check coefficients of fpar <29-01-19, ARL> #

                if self.stationary_flag:
                    if not self.sparseT[i, j]:
                        self.phi1[i, j] = 0.
                        continue
                    else:
                        b_fact = self.LUT[i, j]
                else:
                    b_fact, _ = dblquad(boltz_fact, s1[i] - ds_h, s1[i] + ds_h,
                                        lambda r2: s2[j] - ds_h,
                                        lambda r2: s2[j] + ds_h,
                                        args=b_args, epsabs=0, epsrel=1e-8)

                # Calculate parallel force, then speed of head 1
                fpar = (spring_force(s1[i] - ds_h, s2[j], *f_args) *
                        (-r * a1 + s1[i] - ds_h - s2[j] * b))
                v1a = vhead(vo, fpar, fs)
                fpar = (spring_force(s1[i] + ds_h, s2[j], *f_args) *
                        (-r * a1 + s1[i] + ds_h - s2[j] * b))
                v1b = vhead(vo, fpar, fs)
                # Calculate parallel force, then speed of head 2
                fpar = (spring_force(s1[i], s2[j] - ds_h, *f_args)
                        * (r * a2 + s2[j] - ds_h - s1[i] * b))
                v2a = vhead(vo, fpar, fs)
                fpar = (spring_force(s1[i], s2[j] + ds_h, *f_args)
                        * (r * a2 + s2[j] + ds_h - s1[i] * b))
                v2b = vhead(vo, fpar, fs)
                # Evolve location using first upwind method
                self.phi1[i, j] = self.phi0[i, j] - (
                    CFL * (v1b * self.phi0[i, j] -
                           v1a * self.phi0[i - 1, j] +
                           v2b * self.phi0[i, j] -
                           v2a * self.phi0[i, j - 1]) +
                    dt * ko * (co * b_fact - self.phi0[i, j]))

        # Copy solution n+1 to solution n for next step iteration
        self.phi0 = dcp(self.phi1)

    def Write(self):
        """!TODO: Docstring for Write.
        @return: TODO

        """
        print(self.phi0.tolist())


##########################################
if __name__ == "__main__":
    pdes = PDESolver(sys.argv[1])
    pdes.Run()
