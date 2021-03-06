#!/usr/bin/env python
import time
import numpy as np
# from matplotlib.lines import Line2D

from .analyzer import Analyzer, touch_group, normalize
from .graphs import (pde_graph_all_data_2d, pde_graph_mts_xlink_distr_2d,
                     pde_graph_stationary_runs_2d, pde_graph_moment_data_2d,
                     pde_graph_recreate_xlink_distr_2d)
from .pde_helpers import make_gen_stretch_mat
from .pde_steady_state import pde_steady_state_antipara

"""@package docstring
File: pde_analyzer.py
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
        # What kind of motion of microtubules

        self.s_i = np.asarray(self._h5_data['/rod_data/s1'])
        self.s_j = np.asarray(self._h5_data['/rod_data/s2'])

        if '/OT_data' in self._h5_data:
            self.OT1_pos = self._h5_data['/OT_data/OT1_pos']
            self.OT2_pos = self._h5_data['/OT_data/OT2_pos']
        else:
            self.OT1_pos = None
            self.OT2_pos = None

        self.xl_distr = self._h5_data['/xl_data/xl_distr']
        # self.makexl_densArrs()

        # Max concentration of crosslinkers
        self.max_dens_val = np.amax(self.xl_distr)
        print('Max density: ', self.max_dens_val)

        # Get forces and torques
        self.torque_arr = self._h5_data['/interaction_data/torque_data']
        self.torque_arr = np.linalg.norm(self.torque_arr, axis=2)
        self.force_arr = self._h5_data['/interaction_data/force_data']
        self.force_arr = np.linalg.norm(self.force_arr, axis=2)

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

        xl_analysis_grp = touch_group(analysis_grp, 'xl_analysis')
        self.xl_moment_analysis(xl_analysis_grp, analysis_type)
        self.xl_boundary_analysis(xl_analysis_grp, analysis_type)
        self.xl_work_analysis(xl_analysis_grp, analysis_type)
        self.xl_stretch_distr_analysis(xl_analysis_grp, analysis_type)
        # Analyze error if antiparallel requirements are met
        if (self._params['solver_type'] == 'PDEGenOrientMotorUWSolver' and
                np.dot(self.R1_vec[-1], self.R2_vec[-1]) == -1. and
                np.dot(self.R1_vec[-1], self.R2_pos[-1] - self.R1_pos[-1]) == 0):
            xl_analysis_grp.attrs['error'] = self.xl_measure_error()
            print("Analyzed error: ", xl_analysis_grp.attrs['error'])

        rod_analysis_grp = touch_group(analysis_grp, 'rod_analysis')
        self.rod_geometry_analysis(rod_analysis_grp, analysis_type)

        # if '/OT_data' in self._h5_data:
        # self.ot_analysis()

        t1 = time.time()
        print(("analysis time: {}".format(t1 - t0)))
        return analysis_grp

    def xl_moment_analysis(self, xl_analysis_grp, analysis_type='analyze'):
        """!TODO: Docstring for Momentanalysis.
        @return: TODO

        """
        ds = float(self._params["ds"])
        ds_sqr = ds * ds
        s_i = self.s_i
        s_j = self.s_j

        # Zeroth moment (number of crosslinkers)
        if 'zeroth_moment' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.mu00 = np.sum(self.xl_distr, axis=(0, 1)) * ds_sqr
                self.zero_mom_dset = xl_analysis_grp.create_dataset(
                    'zeroth_moment', data=self.mu00, dtype=np.float32)
            else:
                print('--- The zeroth moment not analyzed or stored. ---')
        else:
            self.zero_mom_dset = xl_analysis_grp['zeroth_moment']
            self.mu00 = np.asarray(self.zero_mom_dset)

        # First moments
        if 'first_moments' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.mu10 = np.einsum('ijn,i->n', self.xl_distr, s_i) * ds_sqr
                self.mu01 = np.einsum('ijn,j->n', self.xl_distr, s_j) * ds_sqr
                self.first_mom_dset = xl_analysis_grp.create_dataset(
                    'first_moments', data=np.stack((self.mu10, self.mu01), axis=-1),
                    dtype=np.float32)
                self.first_mom_dset.attrs['columns'] = ['s_i moment',
                                                        's_j moment']
            else:
                print('--- The first moments not analyzed or stored. ---')
        else:
            self.first_mom_dset = xl_analysis_grp['first_moments']
            self.mu10 = np.asarray(self.first_mom_dset)[:, 0]
            self.mu01 = np.asarray(self.first_mom_dset)[:, 1]

        # Second moments calculations
        if 'second_moments' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.mu11 = np.einsum(
                    'ijn,i,j->n', self.xl_distr, s_i, s_j) * ds_sqr
                self.mu20 = np.einsum(
                    'ijn,i->n', self.xl_distr, s_i * s_i) * ds_sqr
                self.mu02 = np.einsum(
                    'ijn,j->n', self.xl_distr, s_j * s_j) * ds_sqr
                self.second_mom_dset = xl_analysis_grp.create_dataset(
                    'second_moments',
                    data=np.stack((self.mu11, self.mu20, self.mu02), axis=-1),
                    dtype=np.float32)
                self.second_mom_dset.attrs['columns'] = ['s_i*s_j moment',
                                                         's_i^2 moment',
                                                         's_j^2 moment']
            else:
                print('--- The second moments not analyzed or stored. ---')
        else:
            self.second_mom_dset = xl_analysis_grp['second_moments']
            self.mu11 = np.asarray(self.second_mom_dset)[:, 0]
            self.mu20 = np.asarray(self.second_mom_dset)[:, 1]
            self.mu02 = np.asarray(self.second_mom_dset)[:, 2]

    def xl_boundary_analysis(self, xl_analysis_grp, analysis_type='analyze'):
        """!TODO: Docstring for xl_boundary_analysis.
        @return: TODO

        """
        L_i = float(self._params["L1"])
        L_j = float(self._params["L2"])
        ds = float(self._params["ds"])
        ds_sqr = ds * ds
        s_i = self.s_i
        s_j = self.s_j
        # Zeroth boundary term analysis
        if 'zeroth_boundary_terms' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.B0_j = np.sum(self.xl_distr[-2, :, :], axis=0) * ds
                self.B0_i = np.sum(self.xl_distr[:, -2, :], axis=0) * ds
                B0_jp1 = np.sum(self.xl_distr[-1, :, :], axis=0) * ds
                B0_jm1 = np.sum(self.xl_distr[-3, :, :], axis=0) * ds
                B0_ip1 = np.sum(self.xl_distr[:, -1, :], axis=0) * ds
                B0_im1 = np.sum(self.xl_distr[:, -3, :], axis=0) * ds

                self.dBds0_j = (B0_jp1 - B0_jm1) * (.5 / ds)
                self.dBds0_i = (B0_ip1 - B0_im1) * (.5 / ds)
                self.d2Bds0_j = (B0_jp1 - 2. * self.B0_j + B0_jm1) / ds_sqr
                self.d2Bds0_i = (B0_jp1 - 2. * self.B0_j + B0_jm1) / ds_sqr

                self.zeroth_bterm_dset = xl_analysis_grp.create_dataset(
                    'zeroth_boundary_terms',
                    data=np.stack((self.B0_j, self.B0_i,
                                   self.dBds0_j, self.dBds0_i,
                                   self.d2Bds0_j, self.d2Bds0_i), axis=-1),
                    dtype=np.float32)
                self.zeroth_bterm_dset.attrs['columns'] = [
                    'Boundary integral over s_j',
                    'Boundary integral over s_i',
                    'Derivative of boundary term over s_i',
                    'Derivative of boundary term over s_j']
            else:
                print('--- The zeroth boundary term not analyzed or stored. ---')
        else:
            self.zeroth_bterm_dset = xl_analysis_grp['zeroth_boundary_terms']
            self.B0_j = np.asarray(self.zeroth_bterm_dset)[:, 0]
            self.B0_i = np.asarray(self.zeroth_bterm_dset)[:, 1]
            self.dBds0_j = np.asarray(self.zeroth_bterm_dset)[:, 2]
            self.dBds0_i = np.asarray(self.zeroth_bterm_dset)[:, 3]
            self.d2Bds0_j = np.asarray(self.zeroth_bterm_dset)[:, 4]
            self.d2Bds0_i = np.asarray(self.zeroth_bterm_dset)[:, 5]

        if 'first_boundary_terms' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.B1_j = np.einsum('jn,j->n', self.xl_distr[-2], s_j) * ds
                self.B1_i = np.einsum(
                    'in,i->n', self.xl_distr[:, -2], s_i) * ds
                B1_jp1 = np.einsum('jn,j->n', self.xl_distr[-1], s_j) * ds
                B1_jm1 = np.einsum('jn,j->n', self.xl_distr[-3], s_j) * ds
                B1_ip1 = np.einsum('in,i->n', self.xl_distr[:, -1], s_i) * ds
                B1_im1 = np.einsum('in,i->n', self.xl_distr[:, -3], s_i) * ds

                self.dBds1_j = (B1_jp1 - B1_jm1) * (.5 / ds)
                self.dBds1_i = (B1_ip1 - B1_im1) * (.5 / ds)
                self.d2Bds1_j = (B1_jp1 - 2. * self.B1_j + B1_jm1) / ds_sqr
                self.d2Bds1_i = (B1_ip1 - 2. * self.B1_i + B1_im1) / ds_sqr
                self.first_bterm_dset = xl_analysis_grp.create_dataset(
                    'first_boundary_terms',
                    data=np.stack((self.B1_j, self.B1_i,
                                   self.dBds1_j, self.dBds1_i,
                                   self.d2Bds1_j, self.d2Bds1_i), axis=-1),
                    dtype=np.float32)
                self.first_bterm_dset.attrs['columns'] = [
                    'Boundary integral over s_j',
                    'Boundary integral over s_i',
                    'Derivative of boundary term over s_i',
                    'Derivative of boundary term over s_j']
            else:
                print('--- The first boundary term not analyzed or stored. ---')
        else:
            self.first_bterm_dset = xl_analysis_grp['first_boundary_terms']
            self.B1_j = np.asarray(self.first_bterm_dset)[:, 0]
            self.B1_i = np.asarray(self.first_bterm_dset)[:, 1]
            self.dBds1_j = np.asarray(self.first_bterm_dset)[:, 2]
            self.dBds1_i = np.asarray(self.first_bterm_dset)[:, 3]
            self.d2Bds1_j = np.asarray(self.first_bterm_dset)[:, 4]
            self.d2Bds1_i = np.asarray(self.first_bterm_dset)[:, 5]

        if 'second_boundary_terms' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.B2_j = np.einsum(
                    'jn,j->n', self.xl_distr[-1], s_j * s_j) * ds
                self.B2_i = np.einsum(
                    'in,i->n', self.xl_distr[:, -1, :], s_i * s_i) * ds
                B2_jp1 = np.einsum(
                    'jn,j->n', self.xl_distr[-1], s_j * s_j) * ds
                B2_jm1 = np.einsum(
                    'jn,j->n', self.xl_distr[-3], s_j * s_j) * ds
                B2_ip1 = np.einsum(
                    'in,i->n', self.xl_distr[:, -1], s_i * s_i) * ds
                B2_im1 = np.einsum(
                    'in,i->n', self.xl_distr[:, -3], s_i * s_i) * ds

                self.dBds2_j = (B2_jp1 - B2_jm1) * (.5 / ds)
                self.dBds2_i = (B2_ip1 - B2_im1) * (.5 / ds)
                self.d2Bds2_j = (B2_jp1 - 2. * self.B2_j + B2_jm1) / ds_sqr
                self.d2Bds2_i = (B2_ip1 - 2. * self.B2_i + B2_im1) / ds_sqr
                self.second_bterm_dset = xl_analysis_grp.create_dataset(
                    'second_boundary_terms',
                    data=np.stack((self.B2_j, self.B2_i,
                                   self.dBds2_j, self.dBds2_i,
                                   self.d2Bds2_j, self.d2Bds2_i), axis=-1),
                    dtype=np.float32)
                self.second_bterm_dset.attrs['columns'] = [
                    'Boundary integral over s_j',
                    'Boundary integral over s_i',
                    'Derivative of boundary term over s_i',
                    'Derivative of boundary term over s_j']
            else:
                print('--- The second boundary term not analyzed or stored. ---')
        else:
            self.second_bterm_dset = xl_analysis_grp['second_boundary_terms']
            self.B2_j = np.asarray(self.second_bterm_dset)[:, 0]
            self.B2_i = np.asarray(self.second_bterm_dset)[:, 1]
            self.dBds2_j = np.asarray(self.second_bterm_dset)[:, 2]
            self.dBds2_i = np.asarray(self.second_bterm_dset)[:, 3]
            self.d2Bds2_j = np.asarray(self.second_bterm_dset)[:, 4]
            self.d2Bds2_i = np.asarray(self.second_bterm_dset)[:, 5]

        if 'third_boundary_terms' not in xl_analysis_grp:
            if analysis_type != 'load':
                self.B3_j = np.einsum(
                    'jn,j->n', self.xl_distr[-1], s_j * s_j * s_j) * ds
                self.B3_i = np.einsum(
                    'in,i->n', self.xl_distr[:, -1, :], s_i * s_i * s_i) * ds
                B3_jp1 = np.einsum(
                    'jn,j->n', self.xl_distr[-1], s_j * s_j * s_j) * ds
                B3_jm1 = np.einsum(
                    'jn,j->n', self.xl_distr[-3], s_j * s_j * s_j) * ds
                B3_ip1 = np.einsum(
                    'in,i->n', self.xl_distr[:, -1], s_i * s_i * s_i) * ds
                B3_im1 = np.einsum(
                    'in,i->n', self.xl_distr[:, -3], s_i * s_i * s_i) * ds

                self.dBds3_j = (B3_jp1 - B2_jm1) * (.5 / ds)
                self.dBds3_i = (B3_ip1 - B2_im1) * (.5 / ds)
                self.d2Bds3_j = (B3_jp1 - 2. * self.B3_j + B3_jm1) / ds_sqr
                self.d2Bds3_i = (B3_ip1 - 2. * self.B3_i + B3_im1) / ds_sqr
                self.third_bterm_dset = xl_analysis_grp.create_dataset(
                    'third_boundary_terms',
                    data=np.stack((self.B3_j, self.B3_i,
                                   self.dBds3_j, self.dBds3_i,
                                   self.d2Bds3_j, self.d2Bds3_i), axis=-1),
                    dtype=np.float32)
                self.third_bterm_dset.attrs['columns'] = [
                    'Boundary integral over s_j',
                    'Boundary integral over s_i',
                    'Derivative of boundary term over s_i',
                    'Derivative of boundary term over s_j']
            else:
                print('--- The third boundary term not analyzed or stored. ---')
        else:
            self.third_bterm_dset = xl_analysis_grp['third_boundary_terms']
            self.B3_j = np.asarray(self.third_bterm_dset)[:, 0]
            self.B3_i = np.asarray(self.third_bterm_dset)[:, 1]
            self.dBds3_j = np.asarray(self.third_bterm_dset)[:, 2]
            self.dBds3_i = np.asarray(self.third_bterm_dset)[:, 3]
            self.d2Bds3_j = np.asarray(self.third_bterm_dset)[:, 4]
            self.d2Bds3_i = np.asarray(self.third_bterm_dset)[:, 5]

    def xl_stretch_distr_analysis(self, xl_analysis_grp, analysis_type='load'):
        """!TODO: Docstring for xl_stretch_distr_analysis.

        @param xl_analysis_grp: TODO
        @param analysis_type: TODO
        @return: TODO

        """
        nframes = self.time.size
        distr_size = self.xl_distr[:, :, 0].size
        flat_distr_arrs = np.zeros((nframes, distr_size))
        flat_h_arrs = np.zeros((nframes, distr_size))
        s_i = self.s_i
        s_j = self.s_j
        step = .1
        if 'xl_stretch_distr' not in xl_analysis_grp:
            if analysis_type != 'load':
                for t in range(nframes):
                    r_i = self.R1_pos[t]
                    r_j = self.R2_pos[t]
                    u_i = self.R1_vec[t]
                    u_j = self.R2_vec[t]
                    distr = self.xl_distr[:, :, t]

                    stretch_mat = np.linalg.norm(
                        make_gen_stretch_mat(
                            s_i, s_j, u_i, u_j, r_j - r_i, 1.), axis=2)
                    flat_distr_arrs[t] = np.round(distr.flatten(), 9)
                    flat_h_arrs[t] = stretch_mat.flatten()

                flat_h_arrs = np.ma.masked_where(
                    flat_distr_arrs == 0., flat_h_arrs)
                flat_distr_arrs = np.ma.masked_values(flat_distr_arrs, 0)
                max_h = np.amax(flat_h_arrs)
                bin_edges = np.arange(0, max_h + 2 * step, step)
                bin_width = (bin_edges[1] - bin_edges[0])

                self.h_distr = np.zeros((nframes, bin_edges.size - 1))
                for t in range(nframes):
                    self.h_distr[t] = np.histogram(
                        flat_h_arrs[t], bin_edges, weights=flat_distr_arrs[t])[0]
                self.h_distr_dset = xl_analysis_grp.create_dataset(
                    'xl_stretch_distr', data=self.h_distr)
                try:
                    self.h_distr_dset.attrs['bin_edges'] = bin_edges
                except BaseException:
                    pass

                self.h_distr_bin_edges_dset = xl_analysis_grp.create_dataset(
                    'xl_stretch_bin_edges', data=bin_edges)

            else:
                print('--- The stretch distribution not analyzed or stored. ---')
        else:
            self.h_distr_dset = xl_analysis_grp['xl_stretch_distr']

    def xl_work_analysis(self, xl_analysis_grp, analysis_type='analyze'):
        """!TODO: Docstring for xl_work_analysis.

        @param xl_analysis_grp: TODO
        @param analysis_type: TODO
        @return: TODO

        """
        if 'xl_linear_work' not in xl_analysis_grp:
            if analysis_type != 'load':
                # Linear work calculations
                dr_i = np.zeros(self.R1_pos.shape)
                dr_i[1:] = self.R1_pos[1:] - self.R1_pos[:-1]
                f_i = self._h5_data['/interaction_data/force_data'][:, 0, :]
                self.dwl_i = np.zeros(self.R1_pos.shape[0])
                # Use trapezoid rule for numerical integration
                self.dwl_i[1:] = .5 * (np.einsum('ij,ij->i', dr_i[1:], f_i[:-1]) +
                                       np.einsum('ij,ij->i', dr_i[1:], f_i[1:]))

                dr_j = np.zeros(self.R2_pos.shape)
                dr_j[1:] = self.R2_pos[1:] - self.R2_pos[:-1]
                f_j = self._h5_data['/interaction_data/force_data'][:, 1, :]
                self.dwl_j = np.zeros(self.R2_pos.shape[0])
                # Use trapezoid rule for numerical integration
                self.dwl_j[1:] = .5 * (np.einsum('ij,ij->i', dr_j[1:], f_j[:-1]) +
                                       np.einsum('ij,ij->i', dr_j[1:], f_j[1:]))

                # Rotational work calculations
                dtheta_i_vec = np.zeros(self.R1_vec.shape)
                # Get the direction of small rotation
                dtheta_i_vec[1:] = normalize(
                    np.cross(self.R1_vec[:-1], self.R1_vec[1:]))
                # Get amplitude of small rotation
                dtheta_i_vec[1:] *= np.arccos(
                    np.einsum('ij,ij->i', self.R1_vec[1:], self.R1_vec[:-1])
                )[:, None]
                tau_i = self._h5_data['/interaction_data/torque_data'][:, 0, :]
                self.dwr_i = np.zeros(self.R1_vec.shape[0])
                # Use trapezoid rule for numerical integration
                self.dwr_i[1:] = .5 * (
                    np.einsum('ij,ij->i', dtheta_i_vec[1:], tau_i[:-1]) +
                    np.einsum('ij,ij->i', dtheta_i_vec[1:], tau_i[1:]))

                dtheta_j_vec = np.zeros(self.R2_vec.shape)
                dtheta_j_vec[1:] = normalize(
                    np.cross(self.R2_vec[:-1], self.R2_vec[1:]))
                dtheta_j_vec[1:] *= np.arccos(np.einsum('ij,ij->i',
                                                        self.R2_vec[1:],
                                                        self.R2_vec[:-1])
                                              )[:, None]
                tau_j = self._h5_data['/interaction_data/torque_data'][:, 1, :]
                self.dwr_j = np.zeros(self.R2_vec.shape[0])
                self.dwr_j[1:] = .5 * (
                    np.einsum('ij,ij->i', dtheta_j_vec[1:], tau_j[:-1]) +
                    np.einsum('ij,ij->i', dtheta_j_vec[1:], tau_j[1:]))

                self.xl_lin_work_dset = xl_analysis_grp.create_dataset(
                    'xl_linear_work',
                    data=np.stack((self.dwl_i, self.dwl_j), axis=-1),
                    dtype=np.float32)
                self.xl_rot_work_dset = xl_analysis_grp.create_dataset(
                    'xl_rotational_work',
                    data=np.stack((self.dwr_i, self.dwr_j), axis=-1),
                    dtype=np.float32)

            else:
                print('--- The motor work not analyzed or stored. ---')
        else:
            self.xl_lin_work_dset = xl_analysis_grp['xl_linear_work']
            self.xl_rot_work_dset = xl_analysis_grp['xl_rotational_work']
            self.dwl_i = self.xl_lin_work_dset[:, 0]
            self.dwl_j = self.xl_lin_work_dset[:, 1]
            self.dwr_i = self.xl_rot_work_dset[:, 0]
            self.dwr_j = self.xl_rot_work_dset[:, 1]

    def xl_measure_error(self):
        sol = self.xl_distr[:-1, :-1, -1]
        y = np.linalg.norm(self.R2_pos[-1] - self.R1_pos[-1])
        ds = self._params['ds']
        S_i, S_j = np.meshgrid(self.s_i[:-1], self.s_j[:-1], indexing='ij')
        sol_comp = pde_steady_state_antipara(S_i, S_j, y, self._params)

        comp = np.absolute(sol - sol_comp)
        return np.sum(comp) * ds * ds

    def ot_analysis(self):
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

    def graph_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = pde_graph_all_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def graph_reduced_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = pde_graph_mts_xlink_distr_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def graph_orient_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = pde_graph_stationary_runs_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def graph_moment_slice(self, n, fig, axarr):
        """!Graph the solution Psi at a specific time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = pde_graph_moment_data_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts

    def graph_distr_slice(self, n, fig, axarr):
        """!Graph the solution Psi and a recreated distribution at a specific
        time

        @param n: index of slice to graph
        @return: void

        """
        t0 = time.time()
        gca_arts = pde_graph_recreate_xlink_distr_2d(fig, axarr, n, self)
        t1 = time.time()
        print("Graph ", n, "made in: ", t1 - t0)
        return gca_arts
