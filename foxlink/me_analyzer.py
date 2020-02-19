#!/usr/bin/env python

"""@package docstring
File: ME_analysis.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies, and
create graphs from ODE moment expansionruns
"""
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import time

from .analyzer import Analyzer, touch_group

from .graphs import me_graph_all_data_2d


class MEAnalyzer(Analyzer):

    """!Analyze Moment Expansion runs"""

    def __init__(self, filename="Solver.h5", analysis_type='load'):
        """! Initialize analysis code by loading in hdf5 file and setting up
        params.

        @param filename: Name of file to be analyzed
        @param analysis_type: What kind of analysis ot run on data file

        """
        Analyzer.__init__(self, filename, analysis_type)
        self.movie_type = 'all'

    def collect_data_arrays(self):
        """!Store data arrays in member variables
        @return: void, modifies member variables

        """
        Analyzer.collect_data_arrays(self)

        if '/OT_data' in self._h5_data:
            self.OT1_pos = self._h5_data['/OT_data/OT1_pos']
            self.OT2_pos = self._h5_data['/OT_data/OT2_pos']
        else:
            self.OT1_pos = None
            self.OT2_pos = None

        self.mu00 = np.asarray(self._h5_data['/xl_data/zeroth_moment'])
        self.mu10 = np.asarray(self._h5_data['/xl_data/first_moments'][:, 0])
        self.mu01 = np.asarray(self._h5_data['/xl_data/first_moments'][:, 1])
        # self.mu01 = np.asarray(self._h5_data['/xl_data/first_moments'][:, 1])

        # mu_kl = self._h5_data['/xl_data/second_moments']
        mu_kl = self._h5_data['/xl_data/second_moments'][...]
        self.mu11 = mu_kl[:, 0]
        self.mu20 = mu_kl[:, 1]
        self.mu02 = mu_kl[:, 2]

    ########################
    #  Analysis functions  #
    ########################

    def analyze(self, analysis_type='analyze'):
        """!Read in analysis or analyze data according to type of solver hdf5
        file came from and what analysis_type was specified.

        @param analysis_type: load, analyze, overwrite. The extent of the
                              analysis that should be carried out.
        @return: void

        """
        analysis_grp = Analyzer.analyze(self, analysis_type)
        t0 = time.time()

        rod_analysis_grp = touch_group(analysis_grp, 'rod_analysis')
        self.rod_geometry_analysis(rod_analysis_grp)

        interact_analysis_grp = touch_group(analysis_grp,
                                            'interaction_analysis')
        self.force_analysis(interact_analysis_grp)

        # if '/OT_data' in self._h5_data:
        # self.OTAnalysis()

        t1 = time.time()
        print(("Analysis time: {}".format(t1 - t0)))
        return analysis_grp

    def force_analysis(self, interaction_grp, analysis_type='analyze'):
        """!TODO: Docstring for ForceInteractionAnalysis.

        @param grp: TODO
        @return: TODO

        """
        if 'force' not in interaction_grp:
            if analysis_type != 'load':
                ks = self._params['ks']
                self.dR_vec_arr = np.subtract(self.R2_pos, self.R1_pos)
                self.force_vec_arr = -ks * (
                    np.multiply(self.dR_vec_arr, self.mu00[:, None]) +
                    np.multiply(self.mu01[:, None], self.R2_vec) -
                    np.multiply(self.mu10[:, None], self.R1_vec))

                self.force_vec_dset = interaction_grp.create_dataset(
                    'force_vector', data=self.force_vec_arr, dtype=np.float32)

                self.force_arr = np.linalg.norm(self.force_vec_arr, axis=1)
                self.force_mag_dset = interaction_grp.create_dataset(
                    'force_magnitude', data=self.force_arr, dtype=np.float32)
            else:
                print('--- The force on rods not analyzed or stored. ---')
        else:
            self.force_vec_dset = interaction_grp['force_vector']
            self.force_vec_arr = np.asarray(self.force_vec_dset)
            self.force_mag_dset = interaction_grp['force_magnitude']
            self.force_arr = np.asarray(self.force_mag_dset)

    ########################
    #  Graphing functions  #
    ########################

    def graph_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = me_graph_all_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def make_snapshot(self):
        """!Make final snapshot of moment information graph
        @return: void

        """
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        graph_stl = {
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "font.size": 13
        }
        with plt.style.context(graph_stl):
            plt.style.use(graph_stl)
            gs = fig.add_gridspec(2, 3)
            axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                                fig.add_subplot(gs[0, 1]),
                                fig.add_subplot(gs[0, 2]),
                                fig.add_subplot(gs[1, 0]),
                                fig.add_subplot(gs[1, 1]),
                                fig.add_subplot(gs[1, 2]),
                                ])
            fig.suptitle(' ')
            self.graph_slice(-1, fig, axarr)

        fig.savefig('{}_{}.png'.format(self.get_name(), self.graph_type))
