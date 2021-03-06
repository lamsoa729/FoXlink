#!/usr/bin/env python
import numpy as np
from .pde_solver import PDESolver
from .pde_helpers import make_gen_source_mat, make_gen_force_mat, make_gen_torque_mat
from .pde_initial_conditions import *
from .rod_steric_forces import get_min_dist_vec


"""@package docstring
File: pde_gen_orient_solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def reparameterize_rods(R1_pos, R2_pos, R1_vec, R2_vec):
    """!Reparameterize vectors into 4 scalar values

    @param R1_pos: TODO
    @param R2_pos: TODO
    @param R1_vec: TODO
    @param R2_vec: TODO
    @return: 4 parameters of the two tubule state.
            r = distance between rod centers
            a1 = dot product between r-vector and R1_vec
            a2 = dot product between r-vector and R2_vec
            b = dot product between R1_vec and R2_vec

    """
    rvec = R2_pos - R1_pos
    r = np.linalg.norm(rvec)
    if not np.isclose(r, 0., atol=1e-13):
        rvec = rvec / r
        a1 = np.dot(rvec, R1_vec)
        a2 = np.dot(rvec, R2_vec)
    else:
        a1 = 0.
        a2 = 0.
    b = np.dot(R1_vec, R2_vec)
    return r, a1, a2, b


class PDEGenOrientSolver(PDESolver):

    """!Docstring for PDEGenOrientSolver. """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters of PDE system

        @param pfile: TODO
        @param name: TODO

        """
        print("Init PDEGenOrientSolver ->", end=" ")
        PDESolver.__init__(self, pfile=pfile, pdict=pdict)

    def ParseParams(self):
        """! Parse parameters from file and add to member variables
        @return: void
        """
        PDESolver.ParseParams(self)
        # Rod center position vectors of
        self.R1_pos = np.asarray(self._params['R1_pos'])
        self.R2_pos = np.asarray(self._params['R2_pos'])
        print("R1_pos = ", self.R1_pos)
        print("R2_pos = ", self.R2_pos)
        # Rod orientation vectors
        self.R1_vec = np.asarray(self._params['R1_vec'])
        self.R2_vec = np.asarray(self._params['R2_vec'])
        # Make sure to renormalize
        self.R1_vec /= np.linalg.norm(self.R1_vec)
        self.R2_vec /= np.linalg.norm(self.R2_vec)
        if self.steric_flag == 'constrained':
            R1_vec = np.asarray(self._params['R1_vec'])
            R2_vec = np.asarray(self._params['R2_vec'])
            # Make sure to renormalize
            R1_vec /= np.linalg.norm(self.R1_vec)
            R2_vec /= np.linalg.norm(self.R2_vec)
            self.constr_vec = get_min_dist_vec(self.R2_pos - self.R1_pos,
                                               self.R1_vec, self.R2_vec)
            self.constr_vec /= np.linalg.norm(self.constr_vec)

        print("R1_vec = ", self.R1_vec)
        print("R2_vec = ", self.R2_vec)

    def calcSourceMatrix(self):
        """Calculate source matrix for general orientations
        @return: TODO

        """
        r, a1, a2, b = reparameterize_rods(
            self.R1_pos, self.R2_pos, self.R1_vec, self.R2_vec)
        self.src_mat = make_gen_source_mat(self.s1, self.s2, r, a1, a2, b,
                                           self._params['ko'],
                                           self._params['co'],
                                           self._params['ks'],
                                           self._params['ho'],
                                           self._params["beta"])

    def calcForceMatrix(self):
        """!Calculate the force from each crosslinker

        @return: void, changes f_mat, f_mat_dens, force2, force1

        """
        # Create unit direction vector from center of rod1 to center of rod2
        rvec = self.R2_pos - self.R1_pos
        r = np.linalg.norm(rvec)
        if not r == 0:
            rvec /= r

        # Make force density matrix from xlink distributions and stretch
        self.f_mat = make_gen_force_mat(self.s1, self.s2,
                                        self.R1_vec, self.R2_vec,
                                        rvec, r, self._params['ks'],
                                        self._params['ho'])
        self.f_mat_dens = self.sgrid[:, :, None] * self.f_mat[:, :, :]
        # Integrate force density from all xlinks on rods
        self.force2 = (self.f_mat_dens).sum(axis=(0, 1)) * (self.ds ** 2)
        self.force1 = -1. * self.force2

    def calcTorqueMatrix(self):
        """! Calculate the torque provided by each point
        @return: TODO

        """
        # Create torque density matrix
        self.t_mat1 = make_gen_torque_mat(
            -1. * np.swapaxes(self.f_mat_dens, 0, 1), self.s1, self.R1_vec)
        self.t_mat2 = make_gen_torque_mat(
            self.f_mat_dens, self.s2, self.R2_vec)
        # Integrate torque density from all xlinks on rods
        self.torque1 = self.t_mat1.sum(axis=(0, 1)) * self.ds * self.ds
        self.torque2 = self.t_mat2.sum(axis=(0, 1)) * self.ds * self.ds

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: TODO

        """
        PDESolver.makeDataframe(self)
        # Track position and orientations of MTs
        self._R1_pos_dset = self._rod_grp.create_dataset(
            'R1_pos', shape=(self._nframes, 3))
        self._R2_pos_dset = self._rod_grp.create_dataset(
            'R2_pos', shape=(self._nframes, 3))
        self._R1_vec_dset = self._rod_grp.create_dataset(
            'R1_vec', shape=(self._nframes, 3))
        self._R2_vec_dset = self._rod_grp.create_dataset(
            'R2_vec', shape=(self._nframes, 3))
        # Track interactions (forces and torques) between MTs by crosslinkers

    def Write(self):
        """!Write current step in algorithm into dataframe
        @return: TODO

        """
        i_step = PDESolver.Write(self)
        self.GenOrientWrite(i_step)
        return i_step

    def GenOrientWrite(self, i_step):
        self._R1_pos_dset[i_step] = self.R1_pos
        self._R2_pos_dset[i_step] = self.R2_pos
        self._R1_vec_dset[i_step] = self.R1_vec
        self._R2_vec_dset[i_step] = self.R2_vec
