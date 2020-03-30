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

from .graphs import me_graph_all_data_2d, me_graph_distr_data_2d
from .me_zrl_helpers import get_mu_kl_eff


class MEAnalyzer(Analyzer):

    """!Analyze Moment Expansion runs"""

    def __init__(self, filename="Solver.h5", analysis_type='load'):
        """! Initialize analysis code by loading in hdf5 file and setting up
        params.

        @param filename: Name of file to be analyzed
        @param analysis_type: What kind of analysis ot run on data file

        """
        self.xl_distr_flag = False
        Analyzer.__init__(self, filename, analysis_type)
        self.graph_type = 'all'

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

        self.effective_moment_analysis(analysis_grp, analysis_type)

        rod_analysis_grp = touch_group(analysis_grp, 'rod_analysis')
        self.rod_geometry_analysis(rod_analysis_grp)

        interact_analysis_grp = touch_group(analysis_grp,
                                            'interaction_analysis')
        self.force_analysis(interact_analysis_grp)
        self.torque_analysis(interact_analysis_grp)

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
        if 'force_vector' not in interaction_grp:
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

    def torque_analysis(self, interaction_grp, analysis_type='analyze'):
        """!TODO: Docstring for ForceInteractionAnalysis.

        @param grp: TODO
        @return: TODO

        """
        if 'torque_vector' not in interaction_grp:
            if analysis_type != 'load':
                u_i = self.R1_vec
                u_j = self.R2_vec
                r_ij = self.R2_pos - self.R1_pos
                ui_x_uj = np.cross(u_i, u_j, axis=-1)
                ui_x_rij = np.cross(u_i, r_ij, axis=-1)
                uj_x_rij = np.cross(u_j, r_ij, axis=-1)
                ks = self._params['ks']
                # self.dR_vec_arr = np.subtract(self.R2_pos, self.R1_pos)
                torque_i = ks * (self.mu10[:, None] * ui_x_rij +
                                 self.mu11[:, None] * ui_x_uj)
                torque_j = -ks * (self.mu01[:, None] * uj_x_rij +
                                  self.mu11[:, None] * ui_x_uj)
                self.torque_vec_arr = np.stack((torque_i, torque_j), axis=-2)
                self.torque_vec_dset = interaction_grp.create_dataset(
                    'torque_vector', data=self.torque_vec_arr, dtype=np.float32)

                self.torque_arr = np.linalg.norm(self.torque_vec_arr, axis=-1)
                self.torque_mag_dset = interaction_grp.create_dataset(
                    'torque_magnitude', data=self.torque_arr, dtype=np.float32)
            else:
                print('--- The torque on rods not analyzed or stored. ---')
        else:
            self.torque_vec_dset = interaction_grp['torque_vector']
            self.torque_vec_arr = np.asarray(self.torque_vec_dset)
            self.torque_mag_dset = interaction_grp['torque_magnitude']
            self.torque_arr = np.asarray(self.torque_mag_dset)

    def effective_moment_analysis(self, analysis_grp, analysis_type='analyze'):
        """!TODO: Docstring for effective_moment_analysis.

        @param analysis_grp: TODO
        @param analysis_type: TODO
        @return: TODO

        """
        if 'moments_eff' not in analysis_grp:
            if analysis_type != 'load':
                self.mu_kl_eff = np.zeros((self.time.size, 6))
                self._params['L_i'] = self._params['L1']
                self._params['L_j'] = self._params['L2']
                for i in range(self.time.size):
                    mu_kl = [self.mu00[i], self.mu10[i], self.mu01[i],
                             self.mu11[i], self.mu20[i], self.mu02[i]]

                    self.mu_kl_eff[i] = np.asarray(
                        get_mu_kl_eff(mu_kl, self._params))

                self.moments_eff_dset = analysis_grp.create_dataset(
                    'moments_eff', data=self.mu_kl_eff, dtype=np.float32)
            else:
                print('--- The effective motor moments not analyzed or stored. ---')
        else:
            self.moments_eff_dset = analysis_grp['moments_eff']
            self.mu_kl_eff = self.moments_eff_dset[...]

    def make_xl_distr(self):
        """!Make distribution from moment expansion
        @return: TODO

        """
        self.xl_distr_func = self.create_distr_approx_func()
        hL_i = .5 * self._params["L1"]
        hL_j = .5 * self._params["L2"]
        L_i = self._params["L1"]
        L_j = self._params["L2"]
        ds = self._params["ds"]
        ns_i = int(L_i / ds) + 2
        ns_j = int(L_j / ds) + 2
        self.s_i = np.linspace(0, ds * (ns_i - 1), ns_i) - hL_i
        self.s_j = np.linspace(0, ds * (ns_j - 1), ns_j) - hL_j

        s_j_grid, s_i_grid = np.meshgrid(self.s_j, self.s_i)

        self.xl_distr = np.zeros(
            (self.s_i.size, self.s_j.size, self.time.size))
        for i in range(self.time.size):
            self.xl_distr[:, :, i] = self.xl_distr_func(s_i_grid, s_j_grid, i)
        self.max_dens_val = np.amax(self.xl_distr)
        print('Max density: ', self.max_dens_val)
        self.xl_distr_flag = True

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

    def graph_distr_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        if not self.xl_distr_flag:
            self.make_xl_distr()

        t0 = time.time()
        gca_arts = me_graph_distr_data_2d(fig, axarr, n, self)
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
