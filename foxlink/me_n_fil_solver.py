#!/usr/bin/env python
"""
Multi-filament moment expansion solver module.

This module contains solvers for moment expansion equations with multiple filaments,
including both infinite and finite motor pool scenarios.
"""

import numpy as np
from .choose_me_evolver import choose_me_evolver
from .me_solver import MomentExpansionSolver
from .non_dimensionalizer import NonDimensionalizer


class NFilMomentExpansionSolver(MomentExpansionSolver):
    """
    Moment expansion solver for multiple filaments with infinite motor pool.

    This solver handles the dynamics of multiple rigid filaments connected by
    crosslinking motors, assuming an infinite pool of available motors.
    """

    # Override the class-level ODE type
    ODE_type = "n_fil_zrl_xl"  # Default ODE type

    def __init__(self, pfile=None, pdict=None):
        """
        Initialize the N-filament moment expansion solver.

        Args:
            pfile (str, optional): Path to yaml parameter file
            pdict (dict, optional): Parameter dictionary
        """
        print("Init NFilMomentExpansionSolver ->", end=" ")
        super().__init__(pfile, pdict)

    def set_rod_params(self):
        """Extract and set rod parameters from the parameter configuration."""
        self.n_fils = self.rod_arr.shape[0]
        print(f"Number of filaments: {self.n_fils}")

    def set_initial_conditions(self):
        """
        Set the initial state for the solution.

        The solution vector contains:
        - Rod states: 7 * n_fils (position, orientation, length for each rod)
        - Crosslink moments: 2 * n_fils * (n_fils - 1) (mu^kl for each rod pair)
        """
        print("=== Initial conditions ===")

        # Calculate solution vector size
        rod_states_size = 7 * self.n_fils
        xl_moments_size = 2 * self.n_fils * (self.n_fils - 1)
        total_size = rod_states_size + xl_moments_size

        # Initialize solution vector
        self.sol_init = np.zeros(total_size)

        # Set rod initial conditions (flatten rod array)
        self.sol_init[:rod_states_size] = self.rod_arr.flatten()

        # Log initial conditions
        self._log_initial_conditions()

        # Initialize ODE solver
        self.ode_solver = choose_me_evolver(self.sol_init, self)

    def _log_initial_conditions(self):
        """Log the initial conditions for debugging/verification purposes."""
        rod_states_size = 7 * self.n_fils

        print("Rods: (r_x, r_y, r_z, p_x, p_y, p_z, l)")
        rod_states = self.sol_init[:rod_states_size].reshape(-1, 7)
        print(rod_states)

        print("mu_ij^kl: (mu^00, mu^10, mu^01, mu^11)")
        xl_moments = self.sol_init[rod_states_size:].reshape(-1, 4)
        print(xl_moments)

    def make_rod_dataset(self):
        """
        Create HDF5 datasets for rod position, orientation, and length data.

        Extracts rod data from solution and creates separate datasets for:
        - Rod positions (3D coordinates)
        - Rod orientations (unit vectors)
        - Rod lengths (scalar values)
        """
        # Extract and reshape rod data: (n_fils, 7, n_timepoints)
        rod_data = self.sol.y[: self.n_fils * 7].reshape(
            self.n_fils, 7, len(self.t_eval)
        )

        # Create datasets for different rod properties
        self._rod_pos_dset = self._rod_grp.create_dataset(
            "rod_pos",
            data=rod_data[:, :3, :],  # positions (x,y,z)
        )
        self._rod_dir_dset = self._rod_grp.create_dataset(
            "rod_dirs",
            data=rod_data[:, 3:6, :],  # orientations (px,py,pz)
        )
        self._rod_length_dset = self._rod_grp.create_dataset(
            "rod_lengths",
            data=rod_data[:, 6, :],  # lengths
        )

    def make_xl_moment_dataset(self):
        """
        Create HDF5 dataset for crosslink moment data.

        Extracts crosslink moment data from solution and stores as dataset.
        """
        # Extract crosslink moment data starting after rod states
        rod_states_size = self.n_fils * 7
        mu_data = self.sol.y[rod_states_size:].reshape(-1, 4, len(self.t_eval))

        self._mu_dset = self._xl_grp.create_dataset("moments", data=mu_data)

    def non_dimensionalize(self):
        """
        Non-dimensionalize parameters to reduce numerical error in calculations.

        Returns:
            NonDimensionalizer: Object containing dimensionalization information
        """
        # Define non-dimensionalization scales
        # FIXME: Fix the length non-dimensionalization
        non_dim_dict = {
            "time": 1.0,
            "length": 1.0,  # Should be max(L1, L2) or characteristic length
            "energy": 1.0,
        }

        non_dimmer = NonDimensionalizer(**non_dim_dict)

        # Extract and convert rod parameters
        self.rod_arr = np.array(self._params["rods"], dtype=float)

        # Non-dimensionalize rod positions and lengths
        position_length_cols = [0, 1, 2, 6]  # x, y, z, length
        self.rod_arr[:, position_length_cols] = non_dimmer.non_dim_val(
            self.rod_arr[:, position_length_cols], ["length"]
        )

        # Normalize rod orientation vectors
        orientation_cols = slice(3, 6)  # px, py, pz
        norms = np.linalg.norm(self.rod_arr[:, orientation_cols], axis=1)
        self.rod_arr[:, orientation_cols] /= norms[:, np.newaxis]

        # Non-dimensionalize physical parameters
        self._non_dimensionalize_physical_params(non_dimmer)

        return non_dimmer

    def _non_dimensionalize_physical_params(self, non_dimmer):
        """Non-dimensionalize physical parameters using the non-dimensionalizer."""
        self.beta = non_dimmer.non_dim_val(self._params["beta"], ["energy"], [-1])
        self.visc = non_dimmer.non_dim_val(
            self._params["viscosity"], ["energy", "time", "length"], [1, 1, -3]
        )
        self.volume = non_dimmer.non_dim_val(
            float(self._params["volume"]), ["length"], [3]
        )
        self.co = non_dimmer.non_dim_val(float(self._params["co"]), ["length"], [-2])
        self.rod_diam = non_dimmer.non_dim_val(self._params["rod_diameter"], ["length"])

        # Time parameters
        self.dt = non_dimmer.non_dim_val(self.dt, ["time"])
        self.nt = non_dimmer.non_dim_val(self.nt, ["time"])
        self.twrite = non_dimmer.non_dim_val(self.twrite, ["time"])

        # Rate constants
        self.ko = non_dimmer.non_dim_val(self._params["ko"], ["time"], [-1])
        self.ks = non_dimmer.non_dim_val(
            self._params["ks"], ["energy", "length"], [1, -2]
        )

    def redimensionalize(self):
        """Convert results back to dimensional form."""
        # TODO: Implement redimensionalization of results
        pass


class NFilFiniteMomentExpansionSolver(NFilMomentExpansionSolver):
    """
    Moment expansion solver for multiple filaments with finite motor pool.

    Extends the base N-filament solver to handle scenarios with a limited
    number of available crosslinking motors.
    """

    # Override the ODE type for finite crosslinker pool
    ODE_type = "n_fil_finite_zrl_xl"

    def __init__(self, pfile=None, pdict=None):
        """
        Initialize the finite motor pool N-filament moment expansion solver.

        Args:
            pfile (str, optional): Path to yaml parameter file
            pdict (dict, optional): Parameter dictionary
        """
        print("Init NFilFiniteMomentExpansionSolver ->", end=" ")
        super().__init__(pfile, pdict)

    def set_initial_conditions(self):
        """
        Set initial state for finite motor pool solution.

        The solution vector contains:
        - Rod states: 7 * n_fils (position, orientation, length)
        - Crosslink moments: 2 * n_fils * (n_fils - 1) (mu^kl for rod pairs)
        - Unbound motors: 1 (number of unbound motors)
        """
        print("=== Initial conditions ===")

        # Calculate solution vector size (includes unbound motor count)
        rod_states_size = 7 * self.n_fils
        xl_moments_size = 2 * self.n_fils * (self.n_fils - 1)
        total_size = rod_states_size + xl_moments_size + 1  # +1 for unbound motors

        # Initialize solution vector
        self.sol_init = np.zeros(total_size)

        # Set rod initial conditions
        self.sol_init[:rod_states_size] = self.rod_arr.flatten()

        # Set initial number of unbound motors
        self.sol_init[-1] = self._params["n_unbound"]

        # Log initial conditions
        self._log_initial_conditions_finite()

        # Initialize ODE solver
        self.ode_solver = choose_me_evolver(self.sol_init, self)

    def _log_initial_conditions_finite(self):
        """Log initial conditions for finite motor pool case."""
        rod_states_size = 7 * self.n_fils

        print("Rods: (r_x, r_y, r_z, p_x, p_y, p_z, l)")
        rod_states = self.sol_init[:rod_states_size].reshape(-1, 7)
        print(rod_states)

        print("mu_ij^kl: (mu^00, mu^10, mu^01, mu^11)")
        xl_moments = self.sol_init[rod_states_size:-1].reshape(-1, 4)
        print(xl_moments)

        print("Unbound motors:")
        print(self.sol_init[-1])

    def make_xl_moment_dataset(self):
        """
        Create HDF5 datasets for crosslink moments and unbound motor count.

        Creates separate datasets for:
        - Crosslink moment evolution
        - Unbound motor count evolution
        """
        rod_states_size = self.n_fils * 7

        # Extract crosslink moment data (exclude unbound motor count)
        mu_data = self.sol.y[rod_states_size:-1].reshape(-1, 4, len(self.t_eval))
        self._mu_dset = self._xl_grp.create_dataset("moments", data=mu_data)

        # Extract unbound motor count data
        self._unbound_dset = self._xl_grp.create_dataset(
            "unbound", data=self.sol.y[-1, :]
        )
