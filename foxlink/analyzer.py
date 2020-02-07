#!/usr/bin/env python
"""@package docstring
File: analyzer.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing classes to analyze data, make movies,
and create graphs from foxlink runs
"""

from pathlib import Path
import numpy as np
# from matplotlib.lines import Line2D
import h5py
import yaml


def touch_group(parent, grp_name):
    """!See if a data set is there and if it is return it.
        Otherwise, generate it.

    @param parent: Parent group of group to be checked and/or created
    @param grp_name: Name of group to be checked and/or createad
    @return: The group reference

    """
    return parent[grp_name] if grp_name in parent else parent.create_group(
        grp_name)


class Analyzer():

    """!Analyze Fokker-Planck equation code"""

    def __init__(self, filename="Solver.h5", analysis_type='load'):
        """! Initialize analysis code by loading in hdf5 file and setting up
        params.

        @param filename: Name of file to be analyzed
        @param analysis_type: What kind of analysis ot run on data file
        """
        self._filename = filename
        self._h5_data, self._params = self.load()

        self.s_type = self._params['solver_type']

        self.collect_data_arrays()

        self.init_flag = True

        self.analyze(analysis_type)

    def collect_data_arrays(self):
        """!Store data arrays in member variables
        @return: void, modifies member variables

        """
        self.time = np.asarray(self._h5_data["time"])

    def load(self):
        """!Load in data from hdf5 file and grab analysis files if they exist.
        @param analysis_type: load, analyze, overwrite. The extent of the
                              analysis that should be carried out.
        @return: void, stores hdf5 file, parameters, and data arrays to self.

        """
        h5_data = h5py.File(self._filename, 'r+')
        if 'params' in h5_data.attrs:
            params = yaml.safe_load(h5_data.attrs['params'])
        else:
            params = h5_data.attrs
        print(params)
        return h5_data, params

    def save(self):
        """!Create a pickle file of solution
        @return: void

        """
        self._h5_data.flush()
        self._h5_data.close()

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
        if 'analysis' not in self._h5_data:
            if analysis_type == 'load':
                print('-- {} has not been analyzed. --'.format(self._filename))
                return
            self.analysis_grp = self._h5_data.create_group('analysis')
        elif analysis_type == 'overwrite':  # Delete old analysis and try again
            del self._h5_data['analysis']
            self.analysis_grp = self._h5_data.create_group('analysis')
        else:
            self.analysis_grp = self._h5_data['analysis']

        return analysis_grp

    ###########################
    #  Calculation functions  #
    ###########################
    @staticmethod
    def calc_overlap(R1_pos, R2_pos, R1_vec, R2_vec, L1, L2):
        """!Calculate the overlap of two antiparallel rods based on the location
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
        proj = abs(np.einsum('ij,ij->i', R1_vec, R2_vec))
        return proj * (L1 + L2) - dmag

    @staticmethod
    def find_start_time(arr, reps=1):
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
            start_time = Analyzer.find_start_time(arr - arr.mean(), reps - 1)
        else:
            # Get array of sign values, ie. sign with respect to mean
            sign_arr = np.sign(arr)
            # Create array of differences from one index to the next
            diff_arr = np.diff(sign_arr)
            # Find the non-zero differences and record the indices
            index_arr = np.where(diff_arr)[0]  # always produces a tuple
            if index_arr.size == 0:  # System was in steady state all along
                start_time = 0
            else:
                start_time = index_arr[0]
        return start_time
