#!/usr/bin/env python
"""
FoXlink: Framework for crosslink dynamics simulation and analysis.

This module provides the command-line interface for running PDE-based crosslink
dynamics simulations and post-processing analysis workflows.

Author: Adam Lamson
Email: adam.r.lamson@gmail.com
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from matplotlib.animation import FFMpegWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import custom modules with error handling
try:
    from .animation_funcs import (
        make_animation,
        make_minimal_pde_animation,
        make_stat_pde_animation,
        make_distr_pde_animation,
        make_moment_pde_animation,
        make_moment_expansion_animation,
        make_moment_distr_animation,
        make_moment_min_animation,
    )
    from .pde_analyzer import PDEAnalyzer
    from .me_analyzer import MEAnalyzer
    from .profiler import profiler, PROFILING_ENABLED
    from .solver_registry import SolverRegistry

except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


# ================================
# Exception Classes
# ================================


class FoXlinkError(Exception):
    """Base exception for FoXlink errors."""

    pass


class SolverError(FoXlinkError):
    """Errors related to solver operations."""

    pass


class AnalysisError(FoXlinkError):
    """Errors related to analysis operations."""

    pass


class ConfigurationError(FoXlinkError):
    """Errors related to configuration and parameters."""

    pass


# ================================
# Configuration Management
# ================================


class FoXlinkConfig:
    """Configuration container for FoXlink operations."""

    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        self.params: Optional[Dict[str, Any]] = None

    def load_params(self) -> Dict[str, Any]:
        """Load parameters from YAML file with validation."""
        param_file = Path(self.opts.file)

        if not param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")

        try:
            with open(param_file, "r") as f:
                params = yaml.safe_load(f)

            if not isinstance(params, dict):
                raise ValueError("Parameter file must contain a dictionary")

            self.params = params
            logger.debug(f"Loaded {len(params)} parameters from {param_file}")
            return params

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in parameter file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load parameters: {e}")

    def validate_params(self) -> None:
        """Validate required parameters."""
        if not self.params:
            raise ConfigurationError("Parameters not loaded")

        required_keys = ["solver_type"]
        missing_keys = [key for key in required_keys if key not in self.params]

        if missing_keys:
            raise ConfigurationError(f"Missing required parameters: {missing_keys}")

        # Validate solver type exists
        solver_type = self.params["solver_type"]
        available_solvers = SolverRegistry.get_available_solvers()
        if solver_type not in available_solvers:
            raise ConfigurationError(
                f"Unknown solver type: {solver_type}. "
                f"Available solvers: {available_solvers}"
            )


# ================================
# Argument Parsing
# ================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with comprehensive options."""

    parser = argparse.ArgumentParser(
        prog="foxlink",
        description="FoXlink: Framework for crosslink dynamics simulation and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  foxlink -f params.yaml                    # Run simulation
  foxlink -f data.h5 -a analyze -m all     # Analyze and create movie
  foxlink -f data.h5 -a ME -m min          # Moment expansion analysis
  foxlink --list-solvers                   # List available solvers
        """,
    )

    # Input/Output
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=Path("params.yaml"),
        help="Input file (YAML for simulation, HDF5 for analysis)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: same as input file)",
    )

    # Execution modes
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )

    parser.add_argument(
        "-t", "--test", action="store_true", help="Run test protocol (TODO: implement)"
    )

    # Analysis options
    parser.add_argument(
        "-a",
        "--analysis",
        choices=["load", "analyze", "overwrite", "ME"],
        help=(
            "Analysis mode:\n"
            "  load      - Load previously analyzed data\n"
            "  analyze   - Analyze missing data only\n"
            "  overwrite - Re-analyze all data\n"
            "  ME        - Moment expansion analysis"
        ),
    )

    # Visualization options
    parser.add_argument(
        "-m",
        "--movie",
        choices=["all", "min", "stat", "moment", "distr"],
        help=(
            "Create animation:\n"
            "  all    - Complete visualization\n"
            "  min    - Minimal view (rods + crosslinks)\n"
            "  stat   - Statistical view (orientation)\n"
            "  moment - Moment evolution\n"
            "  distr  - Distribution evolution"
        ),
    )

    parser.add_argument(
        "-g",
        "--graph",
        action="store_true",
        help="Generate static plots (TODO: implement for PDE)",
    )

    # Information options
    parser.add_argument(
        "--list-solvers",
        action="store_true",
        help="List all available solver types and exit",
    )

    # Advanced options
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling during simulation"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running simulation/analysis",
    )

    return parser


def validate_arguments(opts: argparse.Namespace) -> None:
    """Validate command-line arguments and set derived options."""

    # Handle list-solvers option
    if opts.list_solvers:
        print("Available solver types:")
        for solver in sorted(SolverRegistry.get_available_solvers()):
            print(f"  {solver}")
        sys.exit(0)

    # Check file exists for analysis mode
    if opts.analysis and not opts.file.exists():
        raise ConfigurationError(f"Input file not found for analysis: {opts.file}")

    # Movie requires analysis
    if opts.movie and not opts.analysis:
        logger.info("Movie generation requires analysis - enabling 'analyze' mode")
        opts.analysis = "analyze"

    # Set logging level based on verbosity
    if opts.verbose >= 3:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif opts.verbose >= 2:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Info logging enabled")
    elif opts.verbose >= 1:
        logging.getLogger().setLevel(logging.WARNING)

    # Validate output directory
    if opts.output:
        opts.output.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory: {opts.output}")


# ================================
# Main FoXlink Controller
# ================================


class FoXlink:
    """Main controller for FoXlink operations."""

    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        self.config = FoXlinkConfig(opts)

    def run_simulation(self) -> None:
        """Execute complete simulation workflow."""

        logger.info("=" * 50)
        logger.info("Starting FoXlink simulation")
        logger.info("=" * 50)

        try:
            # Load and validate parameters
            params = self.config.load_params()
            self.config.validate_params()

            logger.info(f"Loaded parameters from {self.opts.file}")
            logger.info(f"Using solver: {params['solver_type']}")

            # Create solver
            solver = self._create_solver(params)
            logger.info(f"Solver created successfully: {type(solver).__name__}")

            # Run simulation with profiling if requested
            if self.opts.profile and PROFILING_ENABLED:
                logger.info("Profiling enabled")
                profiler.enable()

            try:
                logger.info("Running simulation...")
                solver.run()

                logger.info("Saving results...")
                solver.Save()

                logger.info("✓ Simulation completed successfully")

            finally:
                if self.opts.profile and PROFILING_ENABLED:
                    profiler.disable()
                    logger.info("\nProfiling Results:")
                    profiler.print_stats()

        except Exception as e:
            logger.error(f"✗ Simulation failed: {e}")
            raise SolverError(f"Simulation failed: {e}") from e

    def run_analysis(self) -> None:
        """Execute analysis and visualization workflow."""

        logger.info("=" * 50)
        logger.info("Starting FoXlink analysis")
        logger.info("=" * 50)

        try:
            # Create analyzer
            analyzer = self._create_analyzer()
            logger.info(f"Created analyzer: {type(analyzer).__name__}")

            # Generate movie if requested
            if self.opts.movie:
                self._create_movie(analyzer)

            # Generate plots if requested
            if self.opts.graph:
                self._create_plots(analyzer)

            # Save results
            logger.info("Saving analysis results...")
            analyzer.save()
            logger.info("✓ Analysis completed successfully")

        except Exception as e:
            logger.error(f"✗ Analysis failed: {e}")
            raise AnalysisError(f"Analysis failed: {e}") from e

    def run_dry_run(self) -> None:
        """Validate configuration without running anything."""

        logger.info("Running configuration validation (dry run)")

        try:
            if self.opts.analysis:
                # Validate analysis setup
                analyzer = self._create_analyzer()
                logger.info(
                    f"✓ Analysis configuration valid: {type(analyzer).__name__}"
                )
            else:
                # Validate simulation setup
                params = self.config.load_params()
                self.config.validate_params()
                solver = self._create_solver(params)
                logger.info(
                    f"✓ Simulation configuration valid: {type(solver).__name__}"
                )

            logger.info("✓ All configurations are valid")

        except Exception as e:
            logger.error(f"✗ Configuration validation failed: {e}")
            raise

    def _create_solver(self, params: Dict[str, Any]):
        """Create solver instance from parameters."""

        solver_type = params.get("solver_type")
        if not solver_type:
            raise ConfigurationError("No solver_type specified in parameters")

        try:
            return SolverRegistry.create_solver(solver_type, str(self.opts.file))
        except Exception as e:
            available = SolverRegistry.get_available_solvers()
            raise SolverError(
                f"Failed to create solver '{solver_type}'. "
                f"Available solvers: {available[:5]}..."  # Show first 5 to avoid clutter
            ) from e

    def _create_analyzer(self):
        """Create appropriate analyzer based on analysis mode."""

        file_path = str(self.opts.file)

        if self.opts.analysis == "ME":
            logger.info("Using Moment Expansion analyzer")
            return MEAnalyzer(file_path, "overwrite")
        else:
            analysis_mode = self.opts.analysis or "analyze"
            logger.info(f"Using PDE analyzer in '{analysis_mode}' mode")
            return PDEAnalyzer(file_path, analysis_mode)

    def _create_movie(self, analyzer) -> None:
        """Create animation based on movie type and analyzer."""

        logger.info(f"Creating '{self.opts.movie}' animation...")

        # Configure animation writer
        writer = FFMpegWriter(
            fps=25,
            metadata={
                "artist": "FoXlink",
                "title": f"FoXlink {self.opts.movie} Animation",
            },
            bitrate=1800,
        )

        # Movie function mapping for better organization
        movie_functions = {
            # ME animations (Moment Expansion)
            ("ME", "distr"): make_moment_distr_animation,
            ("ME", "min"): make_moment_min_animation,
            ("ME", "all"): make_moment_expansion_animation,
            ("ME", "moment"): make_moment_expansion_animation,
            ("ME", "stat"): make_moment_expansion_animation,
            # PDE animations
            ("PDE", "all"): make_animation,
            ("PDE", "min"): make_minimal_pde_animation,
            ("PDE", "stat"): make_stat_pde_animation,
            ("PDE", "moment"): make_moment_pde_animation,
            ("PDE", "distr"): make_distr_pde_animation,
        }

        # Determine analyzer type and get appropriate function
        analyzer_type = "ME" if self.opts.analysis == "ME" else "PDE"

        # Set graph type for ME analyzer
        if analyzer_type == "ME":
            analyzer.graph_type = self.opts.movie
            logger.debug(f"Set ME analyzer graph_type to: {self.opts.movie}")

        # Get movie function
        movie_key = (analyzer_type, self.opts.movie)
        movie_func = movie_functions.get(movie_key)

        if not movie_func:
            available_types = [
                key[1] for key in movie_functions.keys() if key[0] == analyzer_type
            ]
            raise AnalysisError(
                f"No movie function for {analyzer_type} analyzer with type '{self.opts.movie}'. "
                f"Available types for {analyzer_type}: {available_types}"
            )

        # Create animation
        try:
            movie_func(analyzer, writer)
            logger.info("✓ Animation created successfully")
        except Exception as e:
            raise AnalysisError(f"Failed to create animation: {e}") from e

    def _create_plots(self, analyzer) -> None:
        """Create static plots based on analyzer type."""

        logger.info("Creating static plots...")

        try:
            if self.opts.analysis == "ME":
                # ME static plotting workflow
                logger.info("Creating ME snapshot plots")
                analyzer.make_snapshot()
                analyzer.init_flag = True
                analyzer.graph_type = "min"
                analyzer.make_snapshot()
                logger.info("✓ ME plots created successfully")
            else:
                # TODO: Implement PDE static plotting
                logger.warning("⚠ PDE static plotting not implemented yet")
                logger.info(
                    "Consider using the movie option (-m) for PDE visualization"
                )

        except Exception as e:
            raise AnalysisError(f"Failed to create plots: {e}") from e


# ================================
# Main Entry Point
# ================================


def main() -> int:
    """
    Main entry point for FoXlink command-line interface.

    Parses arguments, validates configuration, and dispatches to appropriate
    workflow (simulation, analysis, or dry-run).

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """

    try:
        # Parse and validate arguments
        parser = create_argument_parser()
        opts = parser.parse_args()
        validate_arguments(opts)

        # Create FoXlink controller
        foxlink = FoXlink(opts)

        # Dispatch to appropriate workflow
        if opts.dry_run:
            foxlink.run_dry_run()
        elif opts.analysis or opts.movie:
            foxlink.run_analysis()
        else:
            foxlink.run_simulation()

        logger.info("FoXlink completed successfully")
        return 0

    except (ValueError, FileNotFoundError, ConfigurationError) as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FoXlinkError as e:
        logger.error(f"FoXlink error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Show full traceback in verbose mode
        if "opts" in locals() and getattr(opts, "verbose", 0) >= 2:
            import traceback

            traceback.print_exc()
        else:
            logger.info("Use -vv for full traceback")
        return 3


if __name__ == "__main__":
    sys.exit(main())
