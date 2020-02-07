#!/usr/bin/env python
from pathlib import Path
import time
import numpy as np
# from matplotlib.lines import Line2D
import h5py
import yaml

from analyzer import Analyzer, touch_group
from .FP_helpers import *
from .fp_graphs import (fp_graph_all_data_2d, fp_graph_mts_xlink_distr_2d,
                        fp_graph_stationary_runs_2d, fp_graph_moment_data_2d)

"""@package docstring
File: FP_analysis.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies, and create graphs from passive PDE runs
"""


class PDEAnalyzer(Analyzer):

    """!Analyze Fokker-Planck equation code"""

    def __init__(self, filename="Solver.h5", analysis_type='load'):
        """! Initialize analysis code by loading in hdf5 file and setting up
        params.

        @param filename: Name of file to be analyzed
        @param analysis_type: What kind of analysis ot run on data file
        """
        Analyzer.__init__(self, filename, analysis_type)

    def collect_data_arrays(self):
        """!Store data arrays in member variables
        @return: void, modifies member variables

        """
        Analyzer.collect_data_arrays(self)
        self.s1 = np.asarray(self._h5_data['/rod_data/s1'])
        self.s2 = np.asarray(self._h5_data['/rod_data/s2'])
        # What kind of motion of microtubules
        if 'phio' in self._params:  # Ang motion
            self.phi_arr = self._h5_data['rod_data/phi']
        elif 'ro' in self._params:  # Para motion
            self.R_arr = np.asarray(self._h5_data['rod_data/R_pos'])
        else:  # General motion
            self.R1_pos = np.asarray(self._h5_data['/rod_data/R1_pos'])
            self.R2_pos = np.asarray(self._h5_data['/rod_data/R2_pos'])
            self.R1_vec = np.asarray(self._h5_data['/rod_data/R1_vec'])
            self.R2_vec = np.asarray(self._h5_data['/rod_data/R2_vec'])

        if '/OT_data' in self._h5_data:
            self.OT1_pos = self._h5_data['/OT_data/OT1_pos']
            self.OT2_pos = self._h5_data['/OT_data/OT2_pos']
        else:
            self.OT1_pos = None
            self.OT2_pos = None

        self.xl_distr = self._h5_data['/xl_data/xl_distr']
        # self.makexl_densArrs()
        self.Nxl_arr = []  # Array of crosslinker number vs time
        self.torque_arr = []  # Array of torque by crosslinker vs time

        # Max concentration of crosslinkers
        self.max_dens_val = np.amax(self.xl_distr)
        print('Max density: ', self.max_dens_val)

        # Get forces and torques
        self.torque_arr = self._h5_data['/interaction_data/torque_data']
        self.torque_arr = np.linalg.norm(self.torque_arr, axis=2)
        self.force_arr = self._h5_data['/interaction_data/force_data']
        self.force_arr = np.linalg.norm(self.force_arr, axis=2)

    ########################
    #  analysis functions  #
    ########################

    def get_name(self):
        """ Get name of simulation """
        return self._params['name'] if 'name' in self._params else Path.cwd(
        ).name

    def analyze(self, analysis_type='analyze'):
        """!Read in analysis or analyze data according to type of solver hdf5
        file came from and what analysis_type was specified.

        @param analysis_type: load, analyze, overwrite. The extent of the
                              analysis that should be carried out.
        @return: void

        """

        t0 = time.time()

        self.xl_analysis_grp = touch_group(self.analysis_grp, 'xl_analysis')
        self.xl_moment_analysis(self.xl_analysis_grp)

        self.rod_analysis_grp = touch_group(self.analysis_grp, 'rod_analysis')
        self.rod_geometry_analysis(self.rod_analysis_grp)

        # if '/OT_data' in self._h5_data:
        # self.OTanalysis()

        t1 = time.time()
        print(("analysis time: {}".format(t1 - t0)))

        # t2 = time.time()
        # print(("Save time: {}".format(t2 - t1)))

    def xl_moment_analysis(self, xl_analysis_grp, analysis_type='analyze'):
        """!TODO: Docstring for Momentanalysis.
        @return: TODO

        """
        ds = float(self._params["ds"])
        ds_sqr = ds * ds
        s1 = self.s1
        s2 = self.s2

        # Zeroth moment (number of crosslinkers)
        if 'zeroth_moment' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.Nxl_arr = np.sum(self.xl_distr, axis=(0, 1)) * ds_sqr
                self.zero_mom_dset = xl_analysis_grp.create_dataset(
                    'zeroth_moment', data=self.Nxl_arr, dtype=np.float32)
            else:
                print('--- The zeroth moment not analyzed or stored. ---')
        else:
            self.zero_mom_dset = xl_analysis_grp['zeroth_moment']
            self.Nxl_arr = np.asarray(self.zero_mom_dset)

        # First moments
        if 'first_moments' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.P1 = np.einsum('ijn,i->n', self.xl_distr, s1) * ds_sqr
                self.P2 = np.einsum('ijn,j->n', self.xl_distr, s2) * ds_sqr
                self.first_mom_dset = xl_analysis_grp.create_dataset(
                    'first_moments', data=np.stack((self.P1, self.P2), axis=-1),
                    dtype=np.float32)
                self.first_mom_dset.attrs['columns'] = ['s1 moment',
                                                        's2 moment']
            else:
                print('--- The first moments not analyzed or stored. ---')
        else:
            self.first_mom_dset = xl_analysis_grp['first_moments']
            self.P1 = np.asarray(self.first_mom_dset)[:, 0]
            self.P2 = np.asarray(self.first_mom_dset)[:, 1]

        # Second moments calculations
        if 'second_moments' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.u11 = np.einsum(
                    'ijn,i,j->n', self.xl_distr, s1, s2) * ds_sqr
                self.u20 = np.einsum(
                    'ijn,i->n', self.xl_distr, s1 * s1) * ds_sqr
                self.u02 = np.einsum(
                    'ijn,j->n', self.xl_distr, s2 * s2) * ds_sqr
                self.second_mom_dset = xl_analysis_grp.create_dataset(
                    'second_moments',
                    data=np.stack((self.u11, self.u20, self.u02), axis=-1),
                    dtype=np.float32)
                self.second_mom_dset.attrs['columns'] = ['s1*s2 moment',
                                                         's1^2 moment',
                                                         's2^2 moment']
            else:
                print('--- The second moments not analyzed or stored. ---')
        else:
            self.second_mom_dset = xl_analysis_grp['second_moments']
            self.u11 = np.asarray(self.second_mom_dset)[:, 0]
            self.u20 = np.asarray(self.second_mom_dset)[:, 1]
            self.u02 = np.asarray(self.second_mom_dset)[:, 2]

    def rod_geometry_analysis(self, rod_analysis_grp, analysis_type='analyze'):
        """!Analyze and store data relating to the configuration of the rods

        @param rod_analysis_grp: TODO
        @return: TODO

        """
        # Analyze distance between rod center at each time step
        if 'center_separation' not in rod_analysis_grp:
            if analysis_type != 'load':
                self.dR_arr = np.linalg.norm(
                    np.subtract(self.R2_pos, self.R1_pos), axis=1)
                self.rod_sep_dset = rod_analysis_grp.create_dataset(
                    'center_separation', data=self.dR_arr, dtype=np.float32)
            else:
                print('--- The rod center separation not analyzed or stored. ---')
        else:
            self.rod_sep_dset = rod_analysis_grp['center_separation']
            self.dR_arr = np.asarray(self.rod_sep_dset)
        # Analyze angle between rods at teach time step
        if 'angle_between' not in rod_analysis_grp:
            if analysis_type != 'load':
                self.phi_arr = np.arccos(
                    np.einsum('ij,ij->i', self.R1_vec, self.R2_vec))
                self.rod_phi_dset = rod_analysis_grp.create_dataset(
                    'angle_between', data=self.phi_arr, dtype=np.float32)
            else:
                print('--- The angle between rods not analyzed or stored. ---')
        else:
            self.rod_phi_dset = rod_analysis_grp['angle_between']
            self.phi_arr = np.asarray(self.rod_phi_dset)

        # Minus-end(bead) separations
        if 'overlap' not in rod_analysis_grp:
            if analysis_type != 'load':
                self.overlap_arr = Analyzer.calc_overlap(self.R1_pos, self.R2_pos,
                                                         self.R1_vec, self.R2_vec,
                                                         self._params['L1'],
                                                         self._params['L2'])
                self.rod_overlap_dset = rod_analysis_grp.create_dataset(
                    'overlap', data=self.overlap_arr, dtype=np.float32)
            else:
                print('--- The rod overlap not analyzed or stored. ---')
        else:
            self.rod_overlap_dset = rod_analysis_grp['overlap']
            self.overlap_arr = np.asarray(self.rod_phi_dset)

    def OTanalysis(self):
        """!Analyze data for optically trapped rods, especially if they
        are oscillating traps
        @return: void, Adds optical trap post-analysis to hdf5 file

        """
        # Make post processing for optical trap data
        # Get start time by finding when the oscillations first pass mean value
        st = Analyzer.find_start_time(self.overlap_arr, reps=2)
        # TODO Get horizontal separation of optical traps
        ot_sep_arr = np.linalg.norm(self.OT2_pos - self.OT1_pos, axis=1)
        # fft_sep_arr = np.fft.rfft(sep_arr[st:])
        # TODO Get overlap array
        # fft_overlap_arr = np.fft.rfft(overlap_arr[st:])
        # TODO Get horizontal force on rods
        # fft_force_arr = np.fft.rfft(force_arr[st:])
        # TODO Get trap separation
        # TODO Calculate reology components if traps are oscillating


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

    def graphOrientSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = fp_graph_stationary_runs_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def graphMomentSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = fp_graph_moment_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts


##########################################
# if __name__ == "__main__":
    # FPP_analysis = FPanalysis(sys.argv[1])
    # # FPP_analysis.Analyze()
    # FP_analysis.Analyze(True)
    # print("Started making movie")

    # # Movie maker
    # Writer = FFMpegWriter
    # writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)

    # makeAnimation(FPP_analysis, writer)
    # FPP_analysis.Save()
