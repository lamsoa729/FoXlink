#!/usr/bin/env python

"""@package docstring
File: ME_helpers.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def convert_sol_to_geom(sol):
    return (sol[:3], sol[3:6], sol[6:9], sol[9:12])


def sol_print_out(sol):
    """!Print out current solution to solver

    @param r1: Center of mass postion of rod1
    @param r2: Center of mass position of rod2
    @param u1: Orientation unit vector of rod1
    @param u2: Orientation unit vector of rod2
    @param sol: Full solution array of ODE
    @return: void

    """
    r1, r2, u1, u2 = convert_sol_to_geom(sol)
    print ("Step-> r1:", r1, ", r2:", r2, ", u1:", u1, ", u2:", u2)
    print ("       rho:{}, P1:{}, P2:{}, mu11:{}, mu20:{}, mu02:{}".format(
        sol[12], sol[13], sol[14],
        sol[15], sol[16], sol[17]))
