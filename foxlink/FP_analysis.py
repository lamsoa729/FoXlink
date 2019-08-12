#!/usr/bin/env python

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
# from matplotlib.lines import Line2D
import h5py
import yaml

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
    fig = plt.figure(constrained_layout=True, figsize=(15, 13))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
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
        fig.suptitle(' ')
        # FPanal.graphSlice(50, fig, axarr)
        # plt.show()
        nframes = FPanal.time.size
        anim = FuncAnimation(
            fig,
            FPanal.graphSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save('{}.mp4'.format(Path.cwd().name), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def makeMinimalAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 18
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]), ])
        fig.suptitle(' ')
        # FPanal.graphReducedSlice(50, fig, axarr)
        # plt.show()
        nframes = FPanal.time.size
        anim = FuncAnimation(
            fig,
            FPanal.graphReducedSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save('{}_min.mp4'.format(Path.cwd().name), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


class FPAnalysis(object):

    """!Analyze Fokker-Planck equation code"""

    def __init__(self, filename="FP_passive.h5"):
        """! Initialize analysis code by loading in pickle file and setting up
        params.
        """
        self._filename = filename
        self.Load(filename)
        self.time = self._h5_data["time"]
        self.s1 = self._h5_data['MT_data']["s1"]
        self.s2 = self._h5_data['MT_data']["s2"]
        self.sType = self._params['solver_type']
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

        if '/OT_data' in self._h5_data:
            self.OT1_pos = self._h5_data['/OT_data/OT1_pos']
            self.OT2_pos = self._h5_data['/OT_data/OT2_pos']
        else:
            self.OT1_pos = None
            self.OT2_pos = None

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
        if 'params' in self._h5_data.attrs:
            self._params = yaml.safe_load(self._h5_data.attrs['params'])
        else:
            self._params = self._h5_data.attrs
        print(self._params)

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
        self.torque_arr = self._h5_data['/Interaction_data/torque_data']
        self.torque_arr = np.linalg.norm(self.torque_arr, axis=2)
        self.force_arr = self._h5_data['/Interaction_data/force_data']
        self.force_arr = np.linalg.norm(self.force_arr, axis=2)

        self.MT_post_grp = self._h5_data['/MT_data/'].create_group('Post')
        self.XL_post_grp = self._h5_data['/XL_data/'].create_group('Post')
        # Analyze distance between rod center at each time step
        self.dR_arr = np.linalg.norm(np.subtract(self.R2_pos, self.R1_pos),
                                     axis=1)
        self.MT_sep_dset = self.MT_post_grp.create_dataset(
            'center_separation', data=dR_arr, dtype=np.float32)
        # Analyze angle between rods at teach time step
        self.phi_arr = np.arccos(
            np.einsum('ij,ij->i', self.R1_vec, self.R2_vec))
        self.MT_phi_dset = self.MT_post_grp.create_dataset(
            'angle_between', data=phi_arr, dtype=np.float32)

        # Analyze number of crosslinkers at each timestep
        self.Nxl_arr = (np.sum(self.xl_distr, axis=(0, 1)) *
                        (float(self._params["ds"])**2))
        self.Nxl_dset = self.XL_post_grp.create_dataset(
            'xlink_number', data=Nxl_arr, dtype=np.float32)

        # Calculate rod overlap

        L1 = self._params['L1']
        L2 = self._params['L2']
        # Minus-end(bead) separations
        self.overlap = self.calcOverlap(self.R1_pos,
                                        self.R2_pos,
                                        self.R1_vec,
                                        self.R2_vec,
                                        self._params['L1'],
                                        self._params['L2'])

        self.MT_overlap_dset = self.MT_post_grp.create_dataset(
            'overlap', data=overlap, dtype=np.float32)

        if '/OT_data' in self._h5_data:
            self.OTAnalysis()

        t1 = time.time()
        print(("Analysis time: {}".format(t1 - t0)))

        t2 = time.time()
        print(("Save time: {}".format(t2 - t1)))

    def OTAnalysis(self):
        """!Analyze data for optically trapped rods, especially if they
        are oscillating traps
        @return: void, Adds optical trap post-analysis to code

        """
        # Make post processing for optical trap data
        # Get start time by finding when the oscillations first pass mean value
        st = self.FindStartTime(overlap_arr, reps=2)
        # TODO Get horizontal separation of optical traps
        ot_sep_arr = np.linalg.norm(self.OT2_pos - self.OT1_pos, axis=1)
        # fft_sep_arr = np.fft.rfft(sep_arr[st:])
        # TODO Get overlap array
        # fft_overlap_arr = np.fft.rfft(overlap_arr[st:])
        # TODO Get horizontal force on MTs
        # fft_force_arr = np.fft.rfft(force_arr[st:])
        # TODO Get trap separation
        # TODO Calculate reology components if traps are oscillating

###########################
#  Calculation functions  #
###########################
    def calcOverlap(R1_pos, R2_pos, R1_vec, R2_vec, L1, L2):
        """!Calculate the overlap of two antiparalle rods based on the location
        of their minus ends. You can also negate the vector of one of the rods
        if they are parallel instead of antiparallel.

        @param R1_pos: TODO
        @param R2_pos: TODO
        @param R1_vec: TODO
        @param R2_vec: TODO
        @param L1: TODO
        @param L2: TODO
        @return: Overlap of two rods as a function of time

        """
        minus1_pos = R1_pos - .5 * L1 * R1_vec
        minus2_pos = R2_pos - .5 * L2 * R2_vec
        # Distance between beads
        d = np.subtract(minus1_pos, minus2_pos)
        dmag = np.linalg.norm(d, axis=1)
        # Projection of one rod onto another
        proj = abs(np.dot(R1_vec, R2_vec, axis=1))
        return proj * (L1 + L2) - dmag

    @staticmethod
    def FindStartTime(arr, reps=1):
        """! A function to find when simulations reaches a steady state with
        respect to array, arr.

        @param arr: Array to find steady state in
        @param reps: repetitions of recursion
        @return: st Start time, the index of time array when the simulation
        first reaches a the steady state average

        """
        # Test to make sure correct parameters types were given to function
        if not isinstance(arr, np.ndarray):
            raise TypeError(" Array arr must be numpy.ndarray type ")
        if reps > 0:
            st = self.FindStartTime(arr - arr.mean(), reps - 1)
        else:
            # Get array of sign values, ie. sign with respect to mean
            sign_arr = np.sign(arr)
            # Create array of differences from one index to the next
            diff_arr = np.diff(sign_arr)
            # Find the non-zero differences and record the indices
            index_arr = np.where(diff_arr)[0]  # always produces a tuple
            if index_arr.size == 0:  # System was in steady state all along
                st = 0
            else:
                st = index_arr[0]
        return st


########################
#  Graphing functions  #
########################


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

    def graphReducedSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = fp_graph_mts_xlink_distr_2d(fig, axarr, n, self)
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
