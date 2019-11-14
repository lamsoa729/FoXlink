#!/usr/bin/env python
"""@package docstring
File: solver.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: Base Solver class for FoXlink
"""
from pathlib import Path
from copy import deepcopy as dcp
import yaml
import h5py


class Solver():
    """!Abstract class for solver objects. All solving algorithms are implemented
    through these classes.

        Common method functions for inherited solver classes:
            Run <- Abstract = Iterate through time using the Step method
            Step <- Abstract = Implement the specified algorithm at a times step
            setInitialConditions <- Abstract = Set the initial stat of simulation
            makeDataframe <- Abstract = Make a HDF5 data frame
            Write <- Abstract = At a specified time step add current solution to output dataframe.

            ParseParams = Parse parameters from file that is given
            Save = Flush the remaining data in hdf5 file.

        Foxlink uses inheritance to quickly create and combine PDE and ME solving
        algorithms. While solvers do not need to follow specific guidelines, it
        is useful to categorize solvers into different types avoid inheritance
        conflicts when using multi-inheritance. These include:
            Orientation solver -> Defines rods orientations and associate parameters.
                                  eg. Parallel (para),
            Xlink algorithm solver -> What technique is used to solve the xlink
                                      density equations. Can use multiple in one
                                      system. eg. Upwind (UW), Crank-Nicolson(CN)
            Xlink type solver -> What type of crosslinks are being modeled.
                                 eg. Motor, Static, Passive
            Rod algorithm solver -> What technique is used to solve the rods
                                      position equations.
                                      eg. Orient (no motions), Motion, Optical Trap

        Derivative class names are based on their solver type.
    """

    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for PDE to be solved including boundary conditions

        @param pfile: parameter file for PDEs

        """
        print("Init Solver")
        self._pfile = pfile
        self._params = pdict
        self.data_frame_made = False
        self.written = False
        # Initialize parameters from file
        self.ParseParams()
        self.setInitialConditions()
        # self.setBoundaryConditions()

        # Create data frame
        self._h5_fpath = Path("{}.h5".format(self.__class__.__name__))
        self._h5_data = h5py.File(self._h5_fpath, 'w')
        self.makeDataframe()

    def ParseParams(self):
        """! Method to extract and/or calculate parameter values necessary for
        solving algorithms to run.

        @param skip: Load parameter file but skip rest of the parsing. Used
                     when you need to load special parameters before making
                     solution grid in subclasses.

        @return: void, modifies self._params

        """
        if self._pfile is not None:
            with open(self._pfile, 'r') as pf:
                self._params = yaml.safe_load(pf)
        elif self._params is None:
            print("Could not find parameter set.",
                  "Using default params defined in solver.py")
            self._params = dcp(Solver.default_params)

    def Save(self):
        """!Flush and close hdf5 data frame
        @return: void

        """
        self._h5_data.flush()
        self._h5_data.close()

    def makeDataframe(self):
        """! Make data frame to read from later
        @return: void

        """
        # Enter params into hdf5 data file as attributes for later
        self._h5_data.attrs['params'] = yaml.dump(self._params)

    #####################
    #  Virtual methods  #
    #####################

    def setInitialConditions(self):
        """! Set the initial state for the solution.

        @return: void, modifies solution grid

        """
        raise NotImplementedError(
            "Implement setInitialConditions method for {}.".format(
                self.__class__.__name__))

    def Run(self):
        """!Run PDE solver with parameters in pfile through explicity interative time stepping.
        @return: void

        """
        raise NotImplementedError(
            "Implement Run method for {}".format(self.__class__.__name__))

    def Step(self):
        """!Step solver method one unit in time
        @return: None

        """
        raise NotImplementedError("Implement Step method for {}.".format(
            self.__class__.__name__))

    def Write(self):
        """!Write current step in algorithm into data frame
        @return: index of current step

        """
        raise NotImplementedError(
            "Implement Write method for {}.".format(
                self.__class__.__name__))

    default_params = {
        "R1_pos": [0., 0., 0.],
        "R2_pos": [0., 0., 0.],
        "R1_vec": [0., 1., 0.],
        "R2_vec": [0., 1., 0.],
        "L1": 100.,  # Length of microtubule 1
        "L2": 100.,  # Length of microtubule 2
        "rod_diameter": 25,  # Diameter of filament
        "dt": 1.,  # Time step
        "nt": 2000.,  # total time
        "nsteps": 2000,  # total time
        "nwrite": 1,
        "twrite": 1.,
        "ds": 1.,  # Segmentation size of microtubules
        "ko": 1.,  # Crosslinker turnover rate
        "co": 1.,  # Effective crosslinker concentration
        "ks": 1.,  # Crosslinker spring concentration
        "ho": 1.,  # Equilibrium length of crosslinkers
        "vo": 1.,  # Base velocity of crosslinker heads
        "fs": 1.,  # Stall force of crosslinker heads
        "beta": 1.,  # Inverse temperature
        "viscosity": 0.00089,  # Viscosity of fluid filaments are in
        "initial_condition": 'empty',
        "end_pause": False
    }
