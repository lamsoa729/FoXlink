#!/usr/bin/env python

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.lines as lines
import h5py

from .FP_helpers import *
from .fp_graphs import *

"""@package docstring
File: FP_analysis.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies, and create graphs from passive PDE runs
"""


class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''

    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                          'Stdout: {1} StdError: {2}. It may help to re-run '
                          'with --verbose-debug.'.format(e, out, err))


def makeAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    from .stylelib.ase1_styles import ase1_runs_stl
    with plt.style.context(ase1_runs_stl):
        fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
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
    # ani.save('test_movie.mp4', writer=writer)
    # return anim

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
        if hasattr(self, 'phi_arr'):
            self.phi_arr = self._h5_data['MT_data']["phi"]
        elif hasattr(self, 'R_arr'):
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

        self.Nxl_arr = (np.sum(self.xl_distr, axis=(0, 1)) *
                        (float(self._params["ds"])**2))

        t1 = time.time()
        print(("Analysis time: {}".format(t1 - t0)))

        t2 = time.time()
        print(("Save time: {}".format(t2 - t1)))

    def initSlice(self, fig, axarr, c):
        """!Initalize the layout for graphSlice animation

        @param fig: matplotlib figure object
        @param axarr: 2x2 matplotlib axis array
        @return: TODO

        """
        L1 = self._params["L1"]
        L2 = self._params["L2"]
        maxL = max(L1, L2)
        if hasattr(self, 'phi_arr'):
            phi_arr = np.asarray(self.phi_arr)
            max_x = np.amax(maxL * np.cos(.5 * phi_arr))
            max_y = np.amax(maxL * np.sin(.5 * phi_arr))
            min_x = -.1 * max_x
            min_y = -1.1 * max_y
        elif hasattr(self, 'R_arr'):
            x_arr = self.R_arr[:, 0]
            y_arr = self.R_arr[:, 1]
            max_x = np.amax(.5 * L1)
            max_x = max(max_x, np.amax(.5 * L2 + x_arr)) * 1.1
            min_x = np.amin(-.5 * L1)
            min_x = min(min_x, np.amin(-.5 * L2 + x_arr)) * 1.1
            max_y = max(0, np.amax(y_arr)) + .1 * maxL
            min_y = min(0, np.amin(y_arr)) - .1 * maxL
        else:
            r1 = self.R1_pos
            r2 = self.R2_pos
            u1 = self.R1_vec
            u2 = self.R2_vec

            max_x = np.amax(.5 * L1 * u1[:, 1] + r1[:, 1])
            max_x = max(max_x, np.amax(.5 * L2 * u2[:, 1] + r2[:, 1])) * 1.1
            min_x = np.amin(-.5 * L1 * u1[:, 1] + r1[:, 1])
            min_x = min(min_x, np.amin(-.5 * L2 * u2[:, 1] + r2[:, 1])) * 1.1
            max_y = max_x
            min_y = min_x

        axarr[0, 0].set_xlim(min_x, max_x)
        axarr[0, 0].set_ylim(min_y, max_y)
        axarr[0, 0].set_xlabel(r'x (nm)')
        axarr[0, 0].set_ylabel(r'y (nm)')
        fig.colorbar(c, ax=axarr[0, 1])
        axarr[0, 1].set_xlabel(
            'Head distance from MT$_1$ center $s_1$ (nm)')
        axarr[0, 1].set_ylabel(
            'Head distance from MT$_2$ center $s_2$ (nm)')
        axarr[1, 0].set_xlabel(r'Time (sec)')
        axarr[1, 0].set_ylabel(r'Crosslinker torque (pN*nm)')
        axarr[1, 0].set_xlim(left=0, right=self.time[-1])
        axarr[1, 0].set_ylim(np.amin(self.torque_arr),
                             np.amax(self.torque_arr))
        axarr[1, 1].set_xlabel(r'Time (sec)')
        axarr[1, 1].set_ylabel(r'Crosslinker number')
        axarr[1, 1].set_xlim(left=0, right=self.time[-1])
        axarr[1, 1].set_ylim(np.amin(self.Nxl_arr),
                             np.amax(self.Nxl_arr))
        fig.tight_layout()

    def graphSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        # Clean up if lines
        if not self.init_flag:
            for ax in axarr.flatten():
                ax.clear()
            # TODO Check to make sure this didn't fuck things up
            # self.line1.remove()
            # del self.line1
            # self.line2.remove()
            # del self.line2
            for artist in fig.gca().lines + fig.gca().collections:
                artist.remove()
                del artist

        # Draw rods
        L1 = self._params["L1"]
        L2 = self._params["L2"]
        if hasattr(self, 'phi_arr'):
            hphi = self.phi_arr[n] * .5
            self.line1 = lines.Line2D((0, L1 * np.cos(hphi)),
                                      (0, L1 * np.sin(hphi)),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:green', clip_on=False)
            axarr[0, 0].add_line(self.line1)
            self.line2 = lines.Line2D((0, L2 * np.cos(hphi)),
                                      (0, -L2 * np.sin(hphi)),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:purple', clip_on=False)
            axarr[0, 0].add_line(self.line2)
        elif hasattr(self, 'R_arr'):
            r = self.R_arr[n, :]
            self.line1 = lines.Line2D((-.5 * L1, .5 * L1),
                                      (0, 0),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:green', clip_on=False)
            axarr[0, 0].add_line(self.line1)
            self.line2 = lines.Line2D((-.5 * L1 + r[0], .5 * L1 + r[0]),
                                      (r[1], r[1]),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:purple', clip_on=False)
            axarr[0, 0].add_line(self.line2)
        else:
            # r = self.R_arr[n, :]
            r1 = self.R1_pos[n]
            r2 = self.R2_pos[n]
            u1 = self.R1_vec[n]
            u2 = self.R2_vec[n]

            self.line1 = lines.Line2D((-.5 * L1 * u1[1] + r1[1],
                                       .5 * L1 * u1[1] + r1[1]),
                                      (-.5 * L1 * u1[2] + r1[2],
                                       .5 * L1 * u1[2] + r1[2]),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:green', clip_on=False)
            axarr[0, 0].add_line(self.line1)
            self.line2 = lines.Line2D((-.5 * L2 * u2[1] + r2[1],
                                       .5 * L2 * u2[1] + r2[1]),
                                      (-.5 * L2 * u2[2] + r2[2],
                                       .5 * L2 * u2[2] + r2[2]),
                                      linewidth=5, solid_capstyle='round',
                                      color='tab:purple', clip_on=False)
            axarr[0, 0].add_line(self.line2)
            r1 = self.R1_pos
            r2 = self.R2_pos
            u1 = self.R1_vec
            u2 = self.R2_vec

            max_x = np.amax(.5 * L1 * u1[:, 1] + r1[:, 1])
            max_x = max(max_x, np.amax(.5 * L2 * u2[:, 1] + r2[:, 1])) * 1.1
            min_x = np.amin(-.5 * L1 * u1[:, 1] + r1[:, 1])
            min_x = min(min_x, np.amin(-.5 * L2 * u2[:, 1] + r2[:, 1])) * 1.1
            # max_y = np.amax(.5 * L1 * u1[:, 2] + r1[:, 2])
            # max_y = max(max_y, np.amax(.5 * L2 * u2[:, 2] + r2[:, 2])) * 1.1
            # min_y = np.amin(-.5 * L1 * u1[:, 2] + r1[:, 2])
            # min_y = min(min_y, np.amin(-.5 * L2 * u2[:, 2] + r2[:, 2])) * 1.1
            max_y = max_x
            min_y = min_x

            axarr[0, 0].set_xlim(min_x, max_x)
            axarr[0, 0].set_ylim(min_y, max_y)
            axarr[0, 0].set_xlabel(r'x (nm)')
            axarr[0, 0].set_ylabel(r'y (nm)')
            axarr[0, 1].set_xlabel(
                'Head distance from MT$_1$ \n minus-end $s_1$ (nm)')
            axarr[0, 1].set_ylabel(
                'Head distance from MT$_2$ \n minus-end $s_2$ (nm)')
            axarr[1, 0].set_xlabel(r'Time (sec)')
            axarr[1, 0].set_ylabel(r'Crosslinker torque (pN*nm)')
            axarr[1, 0].set_xlim(left=0, right=self.time[-1])
            axarr[1, 0].set_ylim(np.amin(self.torque_arr),
                                 np.amax(self.torque_arr))
            axarr[1, 1].set_xlabel(r'Time (sec)')
            axarr[1, 1].set_ylabel(r'Crosslinker number')
            axarr[1, 1].set_xlim(left=0, right=self.time[-1])
            axarr[1, 1].set_ylim(np.amin(self.Nxl_arr),
                                 np.amax(self.Nxl_arr))

        # Make density plot
        c = graph_xl_dens(axarr[0, 1],
                          self.xl_distr[:, :, n],
                          self.s1,
                          self.s2,
                          max_dens_val=self.max_dens_val)
        if self.init_flag:
            self.initSlice(fig, axarr, c)
            self.init_flag = False
        graph_vs_time(axarr[1, 0], self.time, self.torque_arr, n)
        graph_vs_time(axarr[1, 1], self.time, self.Nxl_arr, n)
        axarr[1, 1].legend(
            ["N({:.2f}) = {:.1f}".format(self.time[n], self.Nxl_arr[n])])
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return fig.gca().lines + fig.gca().collections


##########################################
if __name__ == "__main__":

    FPP_analysis = FPPassiveAnalysis(sys.argv[1])
    # FPP_analysis.Analyze()
    FPP_analysis.Analyze(True)
    print("Started making movie")

    # Movie maker
    # Writer = FasterFFMpegWriter
    Writer = FFMpegWriter
    # print(Writer)
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)

    makeAnimation(FPP_analysis, writer)
    FPP_analysis.Save()
