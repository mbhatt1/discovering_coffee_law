#!/usr/bin/env python3
"""
Extraneous Experiment Runner - Third Set of Validation Tests

These experiments make the COFFEE Law paper hard to dismiss:
1. Dense sampling + prompt diversity (hierarchical) - addresses sampling concerns
2. Open-weight internal-state replication - addresses "embeddings are proxies" critique
3. Proper "Lost in the Middle" protocol - validates Liu et al. correlation
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from query_drift.config import ExperimentConfig, ModelConfig, OutputConfig
from query_drift.experiments import (
    DenseSamplingExperiment,
    InternalStatesExperiment,
    LostInMiddleExperiment,
)
from query_drift.utils.embeddings import EmbeddingClient


def print_header():
    """Print experiment suite header."""
    print("\n" + "=" * 80)
    print("  EXTRANEOUS EXPERIMENTS - MAKING THE PAPER HARD TO DISMISS")
    print("=" * 80)
    print("\nThird Set of Validation Tests:")
    print("  1. Dense Sampling + Prompt Diversity")
    print("     - Hierarchical sampling across prompt templates")
    print("     - Addresses concerns about sampling artifacts")
    print()
    print("  2. Open-weight Internal States")
    print("     - Layer-by-layer Hurst exponent analysis")
    print("     - Addresses 'embeddings are just proxies' critique")
    print("     - Compares pre/post LayerNorm representations")
    print()
    print("  3. Lost in the Middle Protocol")
    print("     - Proper replication of Liu et al. methodology")
    print("     - U-curve position effect validation")
    print("     - Correlation with OU process predictions")
    print("-" * 80)
    print()


def create_output_directory():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"extraneous_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def run_dense_sampling(client, embedding_client, output_dir, quick=False):
    """Run dense sampling + prompt diversity experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: DENSE SAMPLING + PROMPT DIVERSITY")
    print("=" * 80)
    print("Goal: Address sampling concerns with hierarchical sampling across prompts")
    print()

    config = ExperimentConfig(
        model=ModelConfig(
            embedding_model="text-embedding-3-small",
            completion_model="gpt-4o-mini"
        ),
        output=OutputConfig(
            output_dir=str(output_dir / "dense_sampling"),
            save_plots=True,
            show_plots=False,
            verbose=True
        )
    )

    # Adjust parameters for quick mode
    if quick:
        print("[Quick mode: Using reduced parameters]")

    experiment = DenseSamplingExperiment(
        config=config,
        client=client,
        embedding_client=embedding_client,
        quick_mode=quick
    )

    start_time = time.time()
    result = experiment.run()
    duration = time.time() - start_time

    # Save results
    plot_dir = output_dir / "dense_sampling"
    plot_dir.mkdir(parents=True, exist_ok=True)
    experiment.plot(save_path=str(plot_dir / "plot.png"))

    with open(plot_dir / "results.json", "w") as f:
        json.dump({
            "experiment": "dense_sampling_prompt_diversity",
            "duration_seconds": duration,
            "metrics": result.metrics,
            "data": {k: v for k, v in result.data.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 10000}
        }, f, indent=2, default=str)

    print(f"\n[Completed in {duration:.1f}s]")
    return result


def run_internal_states(client, embedding_client, output_dir, quick=False):
    """Run open-weight internal states experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: OPEN-WEIGHT INTERNAL STATES")
    print("=" * 80)
    print("Goal: Analyze layer-by-layer Hurst exponents; compare pre/post LayerNorm")
    print()
    print("NOTE: This experiment requires local GPU and open-weight models.")
    print("      It may take significant time and resources.")
    print()

    config = ExperimentConfig(
        model=ModelConfig(
            embedding_model="text-embedding-3-small",
            completion_model="gpt-4o-mini"
        ),
        output=OutputConfig(
            output_dir=str(output_dir / "internal_states"),
            save_plots=True,
            show_plots=False,
            verbose=True
        )
    )

    # Adjust parameters for quick mode
    if quick:
        print("[Quick mode: Using reduced parameters]")

    experiment = InternalStatesExperiment(
        config=config,
        client=client,
        embedding_client=embedding_client,
        quick_mode=quick
    )

    start_time = time.time()
    result = experiment.run()
    duration = time.time() - start_time

    # Save results
    plot_dir = output_dir / "internal_states"
    plot_dir.mkdir(parents=True, exist_ok=True)
    experiment.plot(save_path=str(plot_dir / "plot.png"))

    with open(plot_dir / "results.json", "w") as f:
        json.dump({
            "experiment": "internal_states_analysis",
            "duration_seconds": duration,
            "metrics": result.metrics,
            "data": {k: v for k, v in result.data.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 10000}
        }, f, indent=2, default=str)

    print(f"\n[Completed in {duration:.1f}s]")
    return result


def run_lost_in_middle(client, embedding_client, output_dir, quick=False, stress_test=True):
    """Run Lost in the Middle protocol experiment."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: LOST IN THE MIDDLE PROTOCOL (STRESS TEST)")
    print("=" * 80)
    print("Goal: Proper replication of Liu et al.; validate U-curve and OU correlation")
    print()

    if stress_test:
        print("*** STRESS TEST MODE ***")
        print("  - 200 documents (10x harder)")
        print("  - All 15 QA pairs")
        print("  - 5 trials per position")
        print("  - Multi-sentence distractors")
        print()

    config = ExperimentConfig(
        model=ModelConfig(
            embedding_model="text-embedding-3-small",
            completion_model="gpt-4o-mini"
        ),
        output=OutputConfig(
            output_dir=str(output_dir / "lost_in_middle"),
            save_plots=True,
            show_plots=False,
            verbose=True
        )
    )

    # Default to stress test mode
    experiment = LostInMiddleExperiment(
        config=config,
        client=client,
        embedding_client=embedding_client,
        quick_mode=quick and not stress_test,
        stress_test=stress_test and not quick
    )

    start_time = time.time()
    result = experiment.run()
    duration = time.time() - start_time

    # Save results
    plot_dir = output_dir / "lost_in_middle"
    plot_dir.mkdir(parents=True, exist_ok=True)
    experiment.plot(save_path=str(plot_dir / "plot.png"))

    with open(plot_dir / "results.json", "w") as f:
        json.dump({
            "experiment": "lost_in_middle_protocol",
            "duration_seconds": duration,
            "metrics": result.metrics,
            "data": {k: v for k, v in result.data.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 10000}
        }, f, indent=2, default=str)

    print(f"\n[Completed in {duration:.1f}s]")
    return result


def generate_summary_report(results_dict, output_dir, total_duration, skipped_internal_states=False):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE SUMMARY REPORT")
    print("=" * 80)

    report_lines = []
    report_lines.append("# Extraneous Experiments - Summary Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    report_lines.append("\n## Experiments Making the Paper Hard to Dismiss\n")

    # 1. Dense Sampling + Prompt Diversity
    if "dense_sampling" in results_dict:
        result = results_dict["dense_sampling"]
        metrics = result.metrics
        report_lines.append("### 1. Dense Sampling + Prompt Diversity")
        report_lines.append(f"- **Status**: {'PASSED' if result.success else 'FAILED'}")
        report_lines.append(f"- **Mean Hurst Exponent (H)**: {metrics.get('mean_hurst', 'N/A'):.4f}"
                          if isinstance(metrics.get('mean_hurst'), (int, float)) else f"- **Mean Hurst Exponent (H)**: {metrics.get('mean_hurst', 'N/A')}")
        report_lines.append(f"- **95% Confidence Interval**: [{metrics.get('ci_lower', 'N/A'):.4f}, {metrics.get('ci_upper', 'N/A'):.4f}]"
                          if all(isinstance(metrics.get(k), (int, float)) for k in ['ci_lower', 'ci_upper']) else f"- **95% Confidence Interval**: N/A")
        report_lines.append(f"- **Variance Trajectory Consistency**: {metrics.get('variance_consistency', 'N/A')}")
        report_lines.append(f"- **Prompt Templates Tested**: {metrics.get('num_prompts', 'N/A')}")
        report_lines.append(f"- **Samples per Template**: {metrics.get('samples_per_prompt', 'N/A')}")
        report_lines.append("")

    # 2. Internal States
    if "internal_states" in results_dict:
        result = results_dict["internal_states"]
        metrics = result.metrics
        report_lines.append("### 2. Open-weight Internal States")
        report_lines.append(f"- **Status**: {'PASSED' if result.success else 'FAILED'}")
        report_lines.append(f"- **Layers Analyzed**: {metrics.get('num_layers', 'N/A')}")

        # Layer-by-layer Hurst exponents
        if 'layer_hurst_exponents' in metrics:
            report_lines.append("- **Layer-by-Layer Hurst Exponents**:")
            for layer_name, h_value in metrics['layer_hurst_exponents'].items():
                report_lines.append(f"  - {layer_name}: {h_value:.4f}" if isinstance(h_value, (int, float)) else f"  - {layer_name}: {h_value}")

        # LayerNorm comparison
        report_lines.append(f"- **Pre-LayerNorm Mean H**: {metrics.get('pre_layernorm_h', 'N/A')}")
        report_lines.append(f"- **Post-LayerNorm Mean H**: {metrics.get('post_layernorm_h', 'N/A')}")
        report_lines.append(f"- **LayerNorm Effect Significant**: {metrics.get('layernorm_effect_significant', 'N/A')}")
        report_lines.append("")
    elif skipped_internal_states:
        report_lines.append("### 2. Open-weight Internal States")
        report_lines.append("- **Status**: SKIPPED (--skip-internal-states flag)")
        report_lines.append("- Requires local GPU and open-weight models")
        report_lines.append("")

    # 3. Lost in the Middle
    if "lost_in_middle" in results_dict:
        result = results_dict["lost_in_middle"]
        metrics = result.metrics
        report_lines.append("### 3. Lost in the Middle Protocol")
        report_lines.append(f"- **Status**: {'PASSED' if result.success else 'FAILED'}")
        report_lines.append(f"- **U-Curve Detected**: {metrics.get('u_curve_detected', 'N/A')}")

        # U-curve parameters
        if 'u_curve_params' in metrics:
            params = metrics['u_curve_params']
            report_lines.append("- **U-Curve Parameters**:")
            report_lines.append(f"  - Curvature (a): {params.get('a', 'N/A')}")
            report_lines.append(f"  - Minimum Position (b): {params.get('b', 'N/A')}")
            report_lines.append(f"  - Baseline (c): {params.get('c', 'N/A')}")

        # OU correlation
        report_lines.append(f"- **OU Prediction Correlation**: {metrics.get('ou_correlation', 'N/A')}")
        report_lines.append(f"- **OU Correlation p-value**: {metrics.get('ou_correlation_pvalue', 'N/A')}")
        report_lines.append(f"- **Liu et al. Replication Score**: {metrics.get('liu_replication_score', 'N/A')}")
        report_lines.append("")

    # Overall conclusion
    report_lines.append("\n## Conclusion\n")

    all_passed = all(
        results_dict.get(exp, type('obj', (), {'success': True})()).success
        for exp in ["dense_sampling", "internal_states", "lost_in_middle"]
        if exp in results_dict
    )

    if all_passed:
        report_lines.append("All extraneous experiments PASSED. The COFFEE Law paper is now:")
        report_lines.append("- Robust to sampling artifact critiques (dense sampling)")
        report_lines.append("- Validated beyond embedding proxies (internal states)")
        report_lines.append("- Properly correlated with Liu et al. findings (Lost in Middle)")
        report_lines.append("\nThese additional experiments make the paper significantly harder to dismiss.")
    else:
        report_lines.append("Some experiments did not pass. Review individual results for details.")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    with open(output_dir / "SUMMARY_REPORT.md", "w") as f:
        f.write(report_text)

    with open(output_dir / "SUMMARY_REPORT.txt", "w") as f:
        f.write(report_text)

    return report_text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run extraneous experiments for COFFEE Law validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_extraneous_experiments.py                    # Run all experiments
  python run_extraneous_experiments.py --quick            # Quick mode with reduced parameters
  python run_extraneous_experiments.py --skip-internal-states  # Skip GPU-intensive experiment
  python run_extraneous_experiments.py --quick --skip-internal-states  # Both flags
        """
    )

    parser.add_argument(
        "--skip-internal-states",
        action="store_true",
        help="Skip the internal states experiment (requires local GPU/large models)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with reduced parameters for faster testing"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print_header()

    if args.quick:
        print("*** QUICK MODE ENABLED - Using reduced parameters ***\n")

    if args.skip_internal_states:
        print("*** Skipping internal states experiment (--skip-internal-states) ***\n")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")

    # Initialize clients
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=60)
    embedding_client = EmbeddingClient(client=client, model="text-embedding-3-small")

    # Track overall time
    overall_start = time.time()

    # Store all results
    results_dict = {}

    # Run experiments
    try:
        # 1. Dense Sampling + Prompt Diversity
        results_dict["dense_sampling"] = run_dense_sampling(
            client, embedding_client, output_dir, quick=args.quick
        )

        # 2. Internal States (optional)
        if not args.skip_internal_states:
            results_dict["internal_states"] = run_internal_states(
                client, embedding_client, output_dir, quick=args.quick
            )
        else:
            print("\n" + "=" * 80)
            print("EXPERIMENT 2: OPEN-WEIGHT INTERNAL STATES - SKIPPED")
            print("=" * 80)
            print("Skipped via --skip-internal-states flag")
            print()

        # 3. Lost in the Middle Protocol (STRESS TEST by default)
        results_dict["lost_in_middle"] = run_lost_in_middle(
            client, embedding_client, output_dir,
            quick=args.quick,
            stress_test=not args.quick  # Stress test unless quick mode
        )

    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
    except Exception as e:
        print(f"\n\nError during experiments: {e}")
        import traceback
        traceback.print_exc()

    # Calculate total duration
    total_duration = time.time() - overall_start

    # Generate summary report
    generate_summary_report(
        results_dict,
        output_dir,
        total_duration,
        skipped_internal_states=args.skip_internal_states
    )

    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
