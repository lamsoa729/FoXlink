#!/usr/bin/env python
from scipy.integrate import dblquad, odeint
import numpy as np
from numba import jit
from .solver import Solver


"""@package docstring
File: ODE_adiabatic_ang_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@jit
def xlink_stretch_ang(s1, s2, phi):
    """!TODO: Docstring for xlink_stretch_ang.

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @return: TODO

    """
    return np.sqrt((s1 * s1) + (s2 * s2) - (2. * s1 * s2 * np.cos(phi)))


@jit
def xlink_avg_torque_ang(s1, s2, phi, co, ks, ho, beta):
    """!TODO: Docstring for xlink_torque_ang.

    @param s1: TODO
    @param s2: TODO
    @param phi: TODO
    @param ks: TODO
    @param ho: TODO
    @return: TODO

    """
    h = xlink_stretch_ang(s1, s2, phi)
    if h == 0:
        return 0
    else:
        return -co * ks * np.sin(phi) * (s1 * s2 * (1. - np.divide(ho, h))
                                         * np.exp(-.5 * beta * ks * np.power(h - ho, 2)))


def total_xlink_torque_ang(L1, L2, phi, co, ks, ho, beta):
    """!TODO: Docstring for xlink_torque_ang.

    @param L1: TODO
    @param L2: TODO
    @param phi: TODO
    @param co: TODO
    @param ks: TODO
    @param ho: TODO
    @param beta: TODO
    @return: TODO

    """
    torque, _ = dblquad(xlink_avg_torque_ang,
                        -.5 * L1, .5 * L1,
                        lambda s2: -.5 * L2, lambda s2: .5 * L2,
                        args=[phi, co, ks, ho, beta],)
    # epsabs=0, epsrel=1.e-8)
    # print("Torque: {}, phi: {}".format(torque, phi))
    return torque


@jit
def rod_rot_mobility(L, diameter, visc):
    """!Find the rotational mobility of a rod using slender body theory

    @param L1: TODO
    @param diameter: TODO
    @param visc: TODO
    @return: TODO

    """
    l = L / diameter
    l2 = l * l

    return ((3. * (l2 * (np.log(l) - .662) + (.917 * l) - .05)) /
            (np.pi * L * L * L * l2 * visc))


def phidot(phi, t, params):
    """!Calculate the derivative of phi at one point in time

    @param phi: TODO
    @param t: TODO
    @paramself._params: TODO
    @return: TODO

    """
    L1, L2, mu, co, ks, ho, beta = self._params
    dphi = mu * total_xlink_torque_ang(L1, L2, phi, co, ks, ho, beta)
    return dphi


class ODEAdiabaticAngSolver(Solver):

    """!Docstring for ODEAdiabaticAngSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!TODO: to be defined1.

        @param pfile: TODO
        @param pdict: TODO

        """
        Solver.__init__(self)

        self._pfile = pfile
        self._params = pdict

    def setInitialConditions(self):
        """!Parse parameters for the run, calculating necessary variables not directly specified including phio and mobility matrices.
        @return: TODO

        """

        L1 = self._params["L1"]
        L2 = self._params["L2"]
        visc = self._params["viscosity"]
        beta = self._params["beta"]
        diameter = self._params["rod_diameter"]
        co = self._params["co"]
        ks = self._params["ks"]
        ho = self._params["ho"]
        # Initial variables
        R1_vec = np.asarray(params['R1_vec'])
        r1_vec = R1_vec / np.linalg.norm(R1_vec)
        R2_vec = np.asarray(params['R2_vec'])
        r2_vec = R2_vec / np.linalg.norm(R2_vec)
        # Calculated variables
        self.phio = np.arccos(np.dot(r1_vec, r2_vec))
        mu1_rot = rod_rot_mobility(L1, diameter, visc)
        mu2_rot = rod_rot_mobility(L2, diameter, visc)
        # Get reduced mobility times 2 since the system experiences. Not sure if
        # this is right.
        mu_eff = 2. * mu1_rot * mu2_rot / (mu1_rot + mu2_rot)
        self.int_params = [L1, L2, mu_eff, co, ks, ho, beta]

    def Run(self):
        """!This uses odeint as its main run functions
        @return: TODO

        """
        self.psoln = odeint(
            phidot, self.phio, self.time, args=(
                self.int_params,))
        self.Write()

    def Write(self):
        """!TODO: Docstring for Write.
        @return: TODO

        """
        # TODO write in psoln, maybe calculate solution grid
        pass

    def Save(self):
        """!TODO: Docstring for Save.
        @return: TODO

        """
        self._h5_data.flush()
        self._h5_data.close()
