#!/usr/bin/env python

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
# from matplotlib.lines import Line2D
import h5py

from .FP_helpers import *
from .fp_graphs import *

"""@package docstring
File: FP_analysis.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies, and create graphs from passive PDE runs
"""


def makeAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    # from .stylelib.ase1_styles import ase1_runs_stl
    fig = plt.figure(constrained_layout=True, figsize=(15, 13))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        # "lines.linewidth": 3,
        # "lines.markersize": 10,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(3, 12)
        axarr = np.asarray([fig.add_subplot(gs[0, :4]),
                            fig.add_subplot(gs[0, 4:8]),
                            fig.add_subplot(gs[0, 8:]),
                            fig.add_subplot(gs[1, :6]),
                            fig.add_subplot(gs[1, 6:]),
                            fig.add_subplot(gs[2, :6]),
                            fig.add_subplot(gs[2, 6:]),
                            ])
        # TODO Change to updated format
        # fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
        nframes = FPanal.time.size
        # nframes = 50
        anim = FuncAnimation(
            fig,
            FPanal.graphSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()
    anim.save('{}.mp4'.format(FPanal.sType), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)

# FIXME: No longer works for FP_pass_CN solvers anymore. Need to re-write
# that code now anyways.


class FPAnalysis(object):

    """!Analyze passive Fokker-Planck equation code"""

    def __init__(self, filename="FP_passive.h5"):
        """! Initialize analysis code by loading in pickle file and setting up
        params.
        """
        self._filename = filename
        self.Load(filename)
        self.time = self._h5_data["time"]
        self.s1 = self._h5_data['MT_data']["s1"]
        self.s2 = self._h5_data['MT_data']["s2"]
        self.sType = self._h5_data.attrs['solver_type']
        # What kind of motion of microtubules
        if 'phio' in self._params:
            self.phi_arr = self._h5_data['MT_data']["phi"]
        elif 'ro' in self._params:
            self.R_arr = np.asarray(self._h5_data['MT_data']["R_pos"])
        else:
            self.R1_pos = self._h5_data['/MT_data/R1_pos']
            self.R2_pos = self._h5_data['/MT_data/R2_pos']
            self.R1_vec = self._h5_data['/MT_data/R1_vec']
            self.R2_vec = self._h5_data['/MT_data/R2_vec']

        self.xl_distr = self._h5_data['/XL_data/XL_distr']
        self.init_flag = True
        # self.makexl_densArrs()
        self.Nxl_arr = []  # Array of crosslinker number vs time
        self.torque_arr = []  # Array of torque by crosslinker vs time

        # Max concentration of crosslinkers
        self.max_dens_val = np.amax(self.xl_distr)
        print('Max density: ', self.max_dens_val)

    def Load(self, filename):
        """!Load in data from hdf5 file
        @param filename: Name of hdf5 file to load
        @return: dictionary of xl_dens

        """
        self._h5_data = h5py.File(filename, 'r+')
        self._params = self._h5_data.attrs

    def Save(self):
        """!Create a pickle file of solution
        @return: void

        """
        self._h5_data.flush()
        self._h5_data.close()

    def Analyze(self, overwrite=False):
        """!TODO: Docstring for Analyze.

        @param overwrite: TODO
        @return: TODO

        """
        t0 = time.time()
        self.torque_arr = np.asarray(
            self._h5_data['/Interaction_data/torque_data'])
        self.torque_arr = np.linalg.norm(self.torque_arr, axis=1)
        self.force_arr = self._h5_data['/Interaction_data/force_data']
        self.force_arr = np.linalg.norm(self.force_arr, axis=1)
        self.dR_arr = np.linalg.norm(np.subtract(self.R2_pos, self.R1_pos),
                                     axis=1)
        self.phi_arr = np.arccos(
            np.einsum('ij,ij->i', self.R1_vec, self.R2_vec))

        self.Nxl_arr = (np.sum(self.xl_distr, axis=(0, 1)) *
                        (float(self._params["ds"])**2))

        t1 = time.time()
        print(("Analysis time: {}".format(t1 - t0)))

        t2 = time.time()
        print(("Save time: {}".format(t2 - t1)))

    def graphSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = fp_graph_all_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts


##########################################
if __name__ == "__main__":

    FPP_analysis = FPAnalysis(sys.argv[1])
    # FPP_analysis.Analyze()
    FP_analysis.Analyze(True)
    print("Started making movie")

    # Movie maker
    # Writer = FasterFFMpegWriter
    Writer = FFMpegWriter
    # print(Writer)
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)

    makeAnimation(FPP_analysis, writer)
    FPP_analysis.Save()