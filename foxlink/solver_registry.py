# ================================
# Solver Registry (Inline)
# ================================


from typing import Optional, Dict, Any, Type, List

# Import all solver classes
# Orientation-based solvers
from .pde_gen_orient_static_xlinks_solver import PDEGenOrientStaticXlinksSolver
from .pde_gen_orient_motor_uw_solver import PDEGenOrientMotorUWSolver

# Free motion solvers
from .pde_gen_motion_static_xlinks_solver import PDEGenMotionStaticXlinksSolver
from .pde_gen_motion_pass_cn_solver import PDEGenMotionPassCNSolver
from .pde_gen_motion_motor_uw_solver import PDEGenMotionMotorUWSolver

# Optical trap solvers
from .pde_ot_gen_motion_motor_uw_solver import PDEOpticalTrapGenMotionMotorUWSolver
from .pde_ot_gen_motion_static_xlinks_solver import (
    PDEOpticalTrapGenMotionStaticXlinksSolver,
)

# Defined motion solvers
from .pde_gen_def_motion_motor_uw_solver import PDEGenDefMotionMotorUWSolver

# Moment expansion solvers
from .me_solver import MomentExpansionSolver
from .me_n_fil_solver import NFilMomentExpansionSolver


class SolverRegistry:
    """Registry for available solver classes."""

    # Central registry of all available solvers
    _solvers: Dict[str, Type] = {
        # Orientation-based solvers
        "PDEGenOrientStaticXlinksSolver": PDEGenOrientStaticXlinksSolver,
        "PDEGenOrientMotorUWSolver": PDEGenOrientMotorUWSolver,
        # Free motion solvers
        "PDEGenMotionStaticXlinksSolver": PDEGenMotionStaticXlinksSolver,
        "PDEGenMotionPassCNSolver": PDEGenMotionPassCNSolver,
        "PDEGenMotionMotorUWSolver": PDEGenMotionMotorUWSolver,
        # Optical trap solvers
        "PDEOpticalTrapGenMotionMotorUWSolver": PDEOpticalTrapGenMotionMotorUWSolver,
        "PDEOpticalTrapGenMotionStaticXlinksSolver": PDEOpticalTrapGenMotionStaticXlinksSolver,
        # Defined motion solvers
        "PDEGenDefMotionMotorUWSolver": PDEGenDefMotionMotorUWSolver,
        # Moment expansion solvers
        "MomentExpansionSolver": MomentExpansionSolver,
        "NFilMomentExpansionSolver": NFilMomentExpansionSolver,
    }

    @classmethod
    def get_solver_class(cls, name: str) -> Type:
        """Get solver class by name."""
        if name not in cls._solvers:
            available = list(cls._solvers.keys())
            raise ValueError(f"Unknown solver: {name}. Available: {available}")
        return cls._solvers[name]

    @classmethod
    def get_available_solvers(cls) -> List[str]:
        """Get list of available solver names."""
        return list(cls._solvers.keys())

    @classmethod
    def create_solver(cls, solver_type: str, param_file: str):
        """Create solver instance."""
        solver_class = cls.get_solver_class(solver_type)
        return solver_class(pfile=param_file)
