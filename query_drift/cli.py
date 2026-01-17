"""
Command-line interface for Query Drift validation experiments.
"""

import argparse
import os
import sys
from pathlib import Path

from .config import ExperimentConfig


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="query-drift",
        description="Empirical validation of the Query Drift Hypothesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m query_drift

  # Run fast preset for quick testing
  python -m query_drift --preset fast

  # Run specific experiments
  python -m query_drift --experiments embedding_drift alignment_decay

  # Use custom config file
  python -m query_drift --config my_config.json

  # Save plots without displaying
  python -m query_drift --save-plots --no-plots
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "--preset",
        choices=["fast", "default", "thorough"],
        default="default",
        help="Configuration preset (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        help="Output directory for results and plots",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot display (still saves if --save-plots is set)",
    )

    plot_save_group = parser.add_mutually_exclusive_group()
    plot_save_group.add_argument(
        "--save-plots",
        action="store_true",
        dest="save_plots",
        help="Save plots to output directory",
    )
    plot_save_group.add_argument(
        "--no-save-plots",
        action="store_false",
        dest="save_plots",
        help="Do not save plots",
    )
    parser.set_defaults(save_plots=None)

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["embedding_drift", "alignment_decay", "loss_scaling", "mem0_retrieval"],
        metavar="EXP",
        help="List of experiments to run (default: all). Choices: embedding_drift, alignment_decay, loss_scaling, mem0_retrieval",
    )

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load or create configuration based on arguments."""
    # Start with preset or config file
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = ExperimentConfig.from_json(config_path)
    elif args.preset == "fast":
        config = ExperimentConfig.fast()
    elif args.preset == "thorough":
        config = ExperimentConfig.thorough()
    else:
        config = ExperimentConfig()

    return config


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration with CLI arguments."""
    # Output directory
    if args.output_dir:
        config.output.output_dir = args.output_dir

    # Plot display
    if args.no_plots:
        config.output.show_plots = False

    # Plot saving
    if args.save_plots is not None:
        config.output.save_plots = args.save_plots

    # Verbosity
    if args.verbose:
        config.output.verbose = True
    elif args.quiet:
        config.output.verbose = False

    # Experiment selection
    if args.experiments:
        # Disable all experiments first
        config.run_embedding_drift = False
        config.run_alignment_decay = False
        config.run_loss_scaling = False
        config.run_mem0_retrieval = False

        # Enable only selected experiments
        for exp in args.experiments:
            if exp == "embedding_drift":
                config.run_embedding_drift = True
            elif exp == "alignment_decay":
                config.run_alignment_decay = True
            elif exp == "loss_scaling":
                config.run_loss_scaling = True
            elif exp == "mem0_retrieval":
                config.run_mem0_retrieval = True

    return config


def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set your OpenAI API key:", file=sys.stderr)
        print("  export OPENAI_API_KEY='your-api-key'", file=sys.stderr)
        return False
    return True


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a summary of the configuration."""
    print("=" * 60)
    print("Query Drift Hypothesis Validation")
    print("=" * 60)
    print()
    print("Experiments to run:")
    if config.run_embedding_drift:
        print(f"  - Embedding Drift (continuations: {config.embedding_drift.num_continuations})")
    if config.run_alignment_decay:
        print(f"  - Alignment Decay (rounds: {config.alignment_decay.num_distractor_rounds})")
    if config.run_loss_scaling:
        print(f"  - Loss Scaling (lengths: {config.loss_scaling.context_lengths})")
    if config.run_mem0_retrieval:
        print(f"  - Mem0 Retrieval (memories: {config.mem0_retrieval.num_memories})")
    print()
    print(f"Output directory: {config.output.output_dir}")
    print(f"Save plots: {config.output.save_plots}")
    print(f"Show plots: {config.output.show_plots}")
    print("=" * 60)
    print()


def print_results_summary(results: dict) -> None:
    """Print a summary of experiment results."""
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    for exp_name, exp_results in results.items():
        print(f"\n{exp_name}:")
        if isinstance(exp_results, dict):
            for key, value in exp_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, (list, tuple)) and len(value) <= 5:
                    print(f"  {key}: {value}")
                elif not isinstance(value, (list, tuple, dict)):
                    print(f"  {key}: {value}")
        else:
            print(f"  {exp_results}")

    print()
    print("=" * 60)


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate environment
    if not validate_environment():
        return 1

    # Load configuration
    config = load_config(args)

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        if "OPENAI_API_KEY" not in warning:  # Already checked
            print(f"Warning: {warning}", file=sys.stderr)

    # Print configuration summary
    if config.output.verbose:
        print_config_summary(config)

    # Check if any experiments are enabled
    if not any([
        config.run_embedding_drift,
        config.run_alignment_decay,
        config.run_loss_scaling,
        config.run_mem0_retrieval,
    ]):
        print("Error: No experiments selected to run.", file=sys.stderr)
        return 1

    # Create output directory
    output_path = Path(config.output.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Import and create validator
    try:
        from .validator import QueryDriftValidator
    except ImportError as e:
        print(f"Error: Failed to import QueryDriftValidator: {e}", file=sys.stderr)
        return 1

    # Run experiments
    try:
        validator = QueryDriftValidator(config)
        results = validator.run_all()
    except Exception as e:
        print(f"Error during experiment execution: {e}", file=sys.stderr)
        if config.output.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Print results summary
    if config.output.verbose:
        print_results_summary(results)

    print("Experiments completed successfully.")
    print(f"Results saved to: {config.output.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
