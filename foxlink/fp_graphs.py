#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml


"""@package docstring
File: fp_graphs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing modular graphing functions for Fokker-Planck data.
"""


def graph_vs_time(ax, time, y, n=-1, color='b'):
    """!TODO: Docstring for graph_vs_t.

    @param ax: TODO
    @param time: TODO
    @param y: TODO
    @param n: TODO
    @return: TODO

    """
    s = ax.scatter(time[:n], y[:n], c=color)
    return s
    # ax.set_xlim(left=0, right=time[-1])


def graph_xl_dens(ax, psi, s1, s2, **kwargs):
    """!Graph an instance in time of the crosslinker density for the FP equation

    @param psi: crosslinker density
    @param **kwargs: TODO
    @return: TODO

    """
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    psi = np.transpose(np.asarray(psi))
    if "max_dens_val" in kwargs:
        max_val = kwargs["max_dens_val"]
        c = ax.pcolormesh(s1, s2, psi, vmin=0, vmax=max_val)
    else:
        c = ax.pcolormesh(s1, s2, psi)
    return c


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
