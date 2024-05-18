#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
from .choose_me_evolver import choose_me_evolver
from .me_solver import MomentExpansionSolver
from .non_dimensionalizer import NonDimensionalizer

# Initial conditions for the system
xlink_number = 100


# Build vector


class NFilMomentExpansionSolver(MomentExpansionSolver):
    def __init__(self, pfile=None, pdict=None):
        """!Set parameters for ODE to be solved including initial conditions.

        @param pfile: yaml parameter file name
        @param pdict: parameter dictionary
        """
        print("Init MomentExpansionSolver ->", end=" ")
        self.ODE_type = "n_fil_zrl_xl"
        MomentExpansionSolver.__init__(self, pfile, pdict)

    def set_rod_params(self):
        # go through parameter file pulling out rod parameters
        self.n_fils = self.rod_arr.shape[0]
        print(f"Number of filaments {self.n_fils}.")

    def set_initial_conditions(self):
        """! Set the initial state for the solution.

        @return: void, modifies solution grid

        """
        print("=== Initial conditions ===")
        self.sol_init = np.zeros(
            7 * self.n_fils + 2 * self.n_fils * (self.n_fils - 1) + 1
        )
        self.sol_init[: 7 * self.n_fils] = self.rod_arr.flatten()
        print("Rods: (r_x, r_y, r_z, p_x, p_y, p_z, l)")
        print(self.sol_init[: 7 * self.n_fils].reshape(-1, 7))

        print("mu_ij^kl: (mu^00, mu^10, mu^01, mu^11)")
        print(self.sol_init[7 * self.n_fils : -1].reshape(-1, 4))

        self.sol_init[-1] = self._params["n_unbound"]
        print("Unbound motors:")
        print(self.sol_init[-1])

        self.ode_solver = choose_me_evolver(self.sol_init, self)

    # def run(self):
    #     print("Run MomentExpansionSolver")

    def make_rod_dataset(self):
        rod_data = self.sol.y[: self.n_fils * 7].reshape(
            self.n_fils, 7, len(self.t_eval)
        )
        self._rod_pos_dset = self._rod_grp.create_dataset(
            "rod_pos", data=rod_data[:, :3, :]
        )
        self._rod_dir_dset = self._rod_grp.create_dataset(
            "rod_dirs", data=rod_data[:, 3:6, :]
        )
        self._rod_length_dset = self._rod_grp.create_dataset(
            "rod_lengths", data=rod_data[:, 6, :]
        )

    def make_xl_moment_dataset(self):
        mu_data = self.sol.y[self.n_fils * 7 : -1].reshape(-1, 4, len(self.t_eval))
        self._mu_dset = self._xl_grp.create_dataset("moments", data=mu_data)
        self._unbound_dset = self._xl_grp.create_dataset(
            "unbound", data=self.sol.y[-1, :]
        )

    def non_dimensionalize(self):
        """!Non-dimensionalize parameters to reduce error in calculations.
        @return: non dimensionalizer

        """
        # FIXME: Fix the length non-dimensionalization
        non_dim_dict = {
            "time": 1.0,
            "length": 1.0,
            # "length": float(max(self._params["L1"], self._params["L2"])),
            "energy": 1.0,
        }
        non_dimmer = NonDimensionalizer(**non_dim_dict)
        # non_dimmer.calc_new_dim('force', ['energy', 'length'], [1, -1])

        self.rod_arr = np.array(self._params["rods"], dtype=float)

        # Non-dimensionalize rods
        self.rod_arr[:, [0, 1, 2, 6]] = non_dimmer.non_dim_val(
            self.rod_arr[:, [0, 1, 2, 6]], ["length"]
        )

        # Normalize rod directions
        self.rod_arr[:, 3:6] /= np.linalg.norm(self.rod_arr[:, 3:6], axis=1)[
            :, np.newaxis
        ]

        self.beta = non_dimmer.non_dim_val(self._params["beta"], ["energy"], [-1])
        self.visc = non_dimmer.non_dim_val(
            self._params["viscosity"], ["energy", "time", "length"], [1, 1, -3]
        )
        self.volume = non_dimmer.non_dim_val(float(self._params["volume"]), ["length"], [3])
        self.co = non_dimmer.non_dim_val(float(self._params["co"]), ["length"], [-2])
        self.rod_diam = non_dimmer.non_dim_val(self._params["rod_diameter"], ["length"])
        self.dt = non_dimmer.non_dim_val(self.dt, ["time"])
        self.nt = non_dimmer.non_dim_val(self.nt, ["time"])
        self.twrite = non_dimmer.non_dim_val(self.twrite, ["time"])
        self.ko = non_dimmer.non_dim_val(self._params["ko"], ["time"], [-1])
        self.ks = non_dimmer.non_dim_val(
            self._params["ks"], ["energy", "length"], [1, -2]
        )
        return non_dimmer

    def redimensionalize(self):
        # TODO Add this back in
        pass
