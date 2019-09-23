#!/usr/bin/env python
from solver.py import Solver
from scipy.integrate import solve_ivp, dblquad
import numpy as np
from numba import jit

"""@package docstring
File: ME_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


@jit
def boltz_fact_zrl(s1, s2, rsqr, a1, a2, b, ks, beta):
    """!TODO: Docstring for boltz_fact_zrl.

    @param s1: TODO
    @param s2: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param ks: TODO
    @param beta: TODO
    @return: TODO

    """
    return np.exp(-.5 * beta * ks * (rsqr + s1**2 + s2**2 -
                                     (2. * s1 * s2 * b) +
                                     2. * (s2 * a2 - s1 * a1)))


def avg_force_zrl(r12, u1, u2, rho, P1, P2, ks):
    """!Find the average force of zero rest length (zrl) crosslinkers on rods

    @param r12: Vector from the center of mass of rod1 to the center of mass of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param ks: motor spring constant
    return: Vector of force from rod1 on rod2

    """
    return -k * (r12 * rho + P2 * u2 - P1 * u1)


def du1_dt_zrl(r12, u1, u2, P1, mu11, a1, b, ks, grot1):
    """!Calculate the time-derivative of rod1's orientation vector with respect
    to the current state of the crosslinked rod system when crosslinkers have
    zero rest length.

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: dot product of r12 and u1
    @param b: dot product of u1 and u2
    @param ks: motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector


    """
    return (k * grot1) * ((r12 - a1) * P1 + (u2 - (b * u1)) * mu11)


def du2_dt_zrl(r12, u1, u2, P2, mu11, a2, b, k, grot1):
    """!Calculate the time-derivative of rod2's orientation vector with respect
    to the current state of the crosslinked rod system when motor have
    zero rest length

    @param r12: Vector from rod1's center of mass to rod2's center of mass
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param mu11: Second motor momement of s1,s2
    @param a1: Dot product of r12 and u1
    @param b: Dot product of u1 and u2
    @param ks: Motor spring constant
    @param grot1: Rotational drag coefficient of rod1
    @return: Time-derivative of rod1's orientation vector


    """
    return (-k * grot2) * ((r12 - a2) * P2 + (u1 - (b * u2)) * mu11)


def drho_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo, fs, ko, c, ks, beta, L1, L2):
    """!Calculate the time-derivative of the zeroth moment of the zero rest length crosslinkers
    bound to the rods

    @param rho: Zeroth motor moment
    @param P1: First motor moment of s1
    @param P2: First motor moment of s2
    @param rsqr: Magnitude squared of the vector from rod1's COM to rod2's COM
    @param a1: Dot product of u1 and r12
    @param a2: Dot product of u2 and r12
    @param b: Dot product of u1 and u2
    @param vo: Velocity of motor when no force is applied
    @param fs: Stall force of motor ends
    @param ks: Motor spring constant
    @param c: Effective concentration of motors in solution
    @param beta: 1/(Boltzmann's constant * Temperature)
    @param L1: Length of rod1
    @param L2: Length of rod2
    @return: Time derivative of the zeroth moment of motors

    """
    # Partition function
    q, e = c * dblquad(boltz_fact_zrl, -.5 * L1, .5 * L1,
                       lambda s2: -.5 * L2, lambda s2: .5 * L2,
                       args=[rsqr, a1, a2, b, ks, beta],)
    # Characteristic walking rate
    kappa = vo * k / fs

    return ((ko * q) + ((vo + kappa * a1) * rho) -
            ((ko + kappa) * P1) + (kappa * b * P2))


def dP1_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo, fs, ko, c, ks, beta):
    """!TODO: Docstring for dP1_dt_zrl.

    @param rho: TODO
    @param P1: TODO
    @param P2: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param vo: TODO
    @param fs: TODO
    @param ko: TODO
    @param c: TODO
    @param ks: TODO
    @param beta: TODO
    @return: TODO

    """
    pass


def dP2_dt_zrl(rho, P1, P2, rsqr, a1, a2, b, vo, fs, ko, c, ks, beta):
    """!TODO: Docstring for dP1_dt_zrl.

    @param rho: TODO
    @param P1: TODO
    @param P2: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param vo: TODO
    @param fs: TODO
    @param ko: TODO
    @param c: TODO
    @param k: TODO
    @param beta: TODO
    @return: TODO

    """
    pass


def dmu11_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta):
    """!TODO: Docstring for dmu11_dt_zrl.

    @param rho: TODO
    @param P1: TODO
    @param P2: TODO
    @param mu11: TODO
    @param mu20: TODO
    @param mu02: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param vo: TODO
    @param fs: TODO
    @param ko: TODO
    @param c: TODO
    @param k: TODO
    @param beta: TODO
    @return: TODO

    """
    pass


def dmu20_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta):
    """!TODO: Docstring for dmu11_dt_zrl.

    @param rho: TODO
    @param P1: TODO
    @param P2: TODO
    @param mu11: TODO
    @param mu20: TODO
    @param mu02: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param vo: TODO
    @param fs: TODO
    @param ko: TODO
    @param c: TODO
    @param k: TODO
    @param beta: TODO
    @return: TODO

    """
    pass


def dmu02_dt_zrl(rho, P1, P2, mu11, mu20, mu02, rsqr,
                 a1, a2, b, vo, fs, ko, c, ks, beta):
    """!TODO: Docstring for dmu11_dt_zrl.

    @param rho: TODO
    @param P1: TODO
    @param P2: TODO
    @param mu11: TODO
    @param mu20: TODO
    @param mu02: TODO
    @param rsqr: TODO
    @param a1: TODO
    @param a2: TODO
    @param b: TODO
    @param vo: TODO
    @param fs: TODO
    @param ko: TODO
    @param c: TODO
    @param k: TODO
    @param beta: TODO
    @return: TODO

    """
    pass


def evolver_zero_rest_length(r1, r2, u1, u2,  # Vectors
                             rho, P1, P2, mu11, mu02, mu20,  # Moments
                             gpara1, gperp1, grot1,  # Friction coefficients
                             gpara2, gperp2, grot2,
                             ks, fs, ko, co, vo, beta, L1, L2):  # Other constants
    """!TODO: Docstring for no_rest_length_evolver.

    @param r1: TODO
    @param r2: TODO
    @param u1: TODO
    @param u2: TODO
    @param gpara1: TODO
    @param gperp1: TODO
    @param grot1: TODO
    @param gpara2: TODO
    @param gperp2: TODO
    @param grot2: TODO
    @param k: TODO
    @param fs: TODO
    @param ko: TODO
    @param co: TODO
    @param vo: TODO
    @param beta: TODO
    @return: TODO

    """
    # Get average force

    pass


class MomentExpansionSolver(Solver):

    """!Solve the evolution of two rods by expanding the Fokker-Planck equation
        in a series of moments of motor end positions on rods.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: TODO
        @param pdict: TODO

        """
        print("Init MomentExpansionSolver -> ")
        Solver.__init__(self, pfile, pdict)

    def setInitialConditions(self):
        """!Set the initial conditions for the system of ODEs
        @return: TODO

        """
        pass

    def ParseParams(self):
        """!Collect parameters from yaml file or dictionary
        @return: TODO

        """
        pass

    def makeDataframe(self):
        """!Create data frame to be written out
        @return: TODO

        """

    def Run(self):
        """!Run algorithm to solve system of ODEs
        @return: TODO

        """
        pass

    def Write(self):
        """!Write out data
        @return: TODO

        """
        pass
