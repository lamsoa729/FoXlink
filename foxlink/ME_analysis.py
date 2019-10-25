#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import h5py
import yaml

from .FP_analysis import FPAnalysis, touchGroup
from .fp_graphs import me_graph_all_data_2d  # TODO change to graph functions

"""@package docstring
File: ME_analysis.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies, and
create graphs from ODE moment expansionruns
"""


class MEAnalysis(FPAnalysis):

    """!Analyze Moment Expansion runs"""

    def __init__(self, filename="Solver.h5", analysis_type='load'):
        """! Initialize analysis code by loading in hdf5 file and setting up
        params.

        @param filename: Name of file to be analyzed
        @param analysis_type: What kind of analysis ot run on data file

        """
        FPAnalysis.__init__(self, filename, analysis_type)

    def collectDataArrays(self):
        """!Store data arrays in member variables
        @return: void, modifies member variables

        """
        self.time = np.asarray(self._h5_data["time"])
        # What kind of motion of microtubules
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

        self.rho = self._h5_data['/XL_data/zeroth_moment']
        self.P_n = self._h5_data['/XL_data/first_moments']
        # TODO get rid of these eventually
        self.P1 = self.P_n[:, 0]
        self.P2 = self.P_n[:, 1]

        self.mu_lk = self._h5_data['/XL_data/second_moments']
        self.u11 = self.mu_lk[:, 0]
        self.u20 = self.mu_lk[:, 1]
        self.u02 = self.mu_lk[:, 2]

    ########################
    #  Analysis functions  #
    ########################

    def Analyze(self, analysis_type='analyze'):
        """!Read in analysis or analyze data according to type of solver hdf5
        file came from and what analysis_type was specified.

        @param analysis_type: load, analyze, overwrite. The extent of the
                              analysis that should be carried out.
        @return: void

        """
        if 'Analysis' not in self._h5_data:
            if analysis_type == 'load':
                print('-- {} has not been analyzed. --'.format(self._filename))
                return
            else:
                self.analysis_grp = self._h5_data.create_group('Analysis')
        elif analysis_type == 'overwrite':  # Delete old analysis and try again
            del self._h5_data['Analysis']
            self.analysis_grp = self._h5_data.create_group('Analysis')
        else:
            self.analysis_grp = self._h5_data['Analysis']

        t0 = time.time()

        # self.XL_analysis_grp = touchGroup(self.analysis_grp, 'XL_analysis')
        # self.XLMomentAnalysis(self.XL_analysis_grp)

        self.rod_analysis_grp = touchGroup(self.analysis_grp, 'rod_analysis')
        self.RodGeometryAnalysis(self.rod_analysis_grp)

        self.interact_analysis_grp = touchGroup(self.analysis_grp,
                                                'interaction_analysis')
        self.ForceAnalysis(self.interact_analysis_grp)

        # if '/OT_data' in self._h5_data:
        # self.OTAnalysis()

        t1 = time.time()
        print(("Analysis time: {}".format(t1 - t0)))

    def ForceAnalysis(self, interaction_grp, analysis_type='analyze'):
        """!TODO: Docstring for ForceInteractionAnalysis.

        @param grp: TODO
        @return: TODO

        """
        if 'force' not in interaction_grp:
            if analysis_type != 'load':
                ks = self._params['ks']
                self.dR_arr = np.subtract(self.R2_pos, self.R1_pos)
                self.force_vec_arr = avg_force_zrl(
                    r12, self.R1_vec, self.R2_vec, self.P1, self.P2, ks)
                self.force_vec_dset = interaction_grp.create_dset(
                    'force_vector', data=self.force12, dtype=np.float32)

                self.force_arr = np.linalg.norm(force12, axis=1)
                self.force_mag_dset = interaction_grp.create_dset(
                    'force_magnitude', data=self.force_arr, dtype=np.float32)
            else:
                print('--- The force on rods not analyzed or stored. ---')
        else:
            self.force_vec_dset = interaction_grp['force_vector']
            self.force_vec_arr = np.asarray(self.force_vec_dset)
            self.force_mag_dset = interaction_grp['force_magnitude']
            self.force_arr = np.asarray(self.force_mag_dset)

            # self.dR_arr = np.asarray(self.rod_sep_dset)

    ########################
    #  Graphing functions  #
    ########################

    def graphSlice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = me_graph_all_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts
