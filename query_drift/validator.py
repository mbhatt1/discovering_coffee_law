"""
Query Drift Validator - Main orchestrator for all experiments.

This module provides the QueryDriftValidator class that coordinates
the execution of all four Query Drift Hypothesis validation experiments.
"""

import os
import json
import time
from typing import Optional

from openai import OpenAI

from .config import ExperimentConfig
from .experiments import (
    EmbeddingDriftExperiment,
    AlignmentDecayExperiment,
    LossScalingExperiment,
    Mem0RetrievalExperiment,
)
from .visualization import ResultVisualizer


class QueryDriftValidator:
    """
    Orchestrates all Query Drift Hypothesis validation experiments.

    This class manages the execution of four experiments that test
    different predictions of the Query Drift Hypothesis:
    1. Embedding Drift - Tests embedding similarity decay over token generation
    2. Alignment Decay - Tests attention alignment decay with distractors
    3. Loss Scaling - Tests loss scaling with context length
    4. Mem0 Retrieval - Tests memory retrieval degradation over time
    """

    # Expected exponent values for hypothesis validation
    EXPECTED_H = 0.5  # Expected Hurst exponent
    EXPECTED_BETA = 0.5  # Expected decay exponent
    TOLERANCE = 0.15  # Tolerance for prediction validation

    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize the QueryDriftValidator.

        Args:
            config: Experiment configuration. If None, uses default configuration.
        """
        self.config = config or ExperimentConfig()

        # Create OpenAI client
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=self.config.model.api_timeout
        )

        # Initialize all experiments
        self.embedding_drift = EmbeddingDriftExperiment(
            config=self.config,
            client=self.client
        )
        self.alignment_decay = AlignmentDecayExperiment(
            config=self.config,
            client=self.client
        )
        self.loss_scaling = LossScalingExperiment(
            config=self.config,
            client=self.client
        )
        self.mem0_retrieval = Mem0RetrievalExperiment(
            config=self.config,
            client=self.client
        )

        # Create visualizer
        self.visualizer = ResultVisualizer(
            output_dir=self.config.output.output_dir,
            save_plots=self.config.output.save_plots,
            show_plots=self.config.output.show_plots,
            dpi=self.config.output.plot_dpi
        )

        # Results storage
        self.results = {}
        self.summary = {}

    def run_all(self) -> dict:
        """
        Run all configured experiments and compile results.

        Returns:
            Dictionary containing all experiment results and summary.
        """
        self._print_header()

        start_time = time.time()

        # Run each experiment based on config flags
        if self.config.run_embedding_drift:
            self._print_experiment_start("Embedding Drift")
            self.results["embedding_drift"] = self.embedding_drift.run()
            self._print_experiment_complete("Embedding Drift")

        if self.config.run_alignment_decay:
            self._print_experiment_start("Alignment Decay")
            self.results["alignment_decay"] = self.alignment_decay.run()
            self._print_experiment_complete("Alignment Decay")

        if self.config.run_loss_scaling:
            self._print_experiment_start("Loss Scaling")
            self.results["loss_scaling"] = self.loss_scaling.run()
            self._print_experiment_complete("Loss Scaling")

        if self.config.run_mem0_retrieval:
            self._print_experiment_start("Mem0 Retrieval")
            self.results["mem0_retrieval"] = self.mem0_retrieval.run()
            self._print_experiment_complete("Mem0 Retrieval")

        total_time = time.time() - start_time

        # Compile summary
        self.summary = self._compile_summary()
        self.summary["total_time_seconds"] = total_time

        # Save results to JSON
        if self.config.output.save_json:
            self._save_results()

        # Generate plots
        if self.config.output.save_plots or self.config.output.show_plots:
            self._generate_plots()

        # Print summary
        self._print_summary()

        return {
            "results": self.results,
            "summary": self.summary
        }

    def _compile_summary(self) -> dict:
        """
        Compile summary of all experiment results.

        Returns:
            Dictionary containing exponents and prediction validations.
        """
        summary = {
            "exponents": {},
            "predictions": {},
            "all_passed": True
        }

        def get_metric(result, key):
            """Extract metric from ExperimentResult or dict."""
            if hasattr(result, 'metrics'):
                return result.metrics.get(key)
            elif isinstance(result, dict):
                return result.get(key) or result.get('metrics', {}).get(key)
            return None

        # Collect exponents from each experiment
        if "embedding_drift" in self.results:
            result = self.results["embedding_drift"]
            h_exp = get_metric(result, "hurst_exponent")
            if h_exp is not None:
                summary["exponents"]["embedding_drift_H"] = h_exp
                passed = abs(h_exp - self.EXPECTED_H) <= self.TOLERANCE
                summary["predictions"]["embedding_drift"] = {
                    "expected": self.EXPECTED_H,
                    "observed": h_exp,
                    "passed": passed,
                    "description": f"H ~ {self.EXPECTED_H} (Brownian motion)"
                }
                if not passed:
                    summary["all_passed"] = False

        if "alignment_decay" in self.results:
            result = self.results["alignment_decay"]
            beta = get_metric(result, "decay_exponent")
            if beta is not None:
                summary["exponents"]["alignment_decay_beta"] = beta
                passed = abs(beta - self.EXPECTED_BETA) <= self.TOLERANCE
                summary["predictions"]["alignment_decay"] = {
                    "expected": self.EXPECTED_BETA,
                    "observed": beta,
                    "passed": passed,
                    "description": f"beta ~ {self.EXPECTED_BETA} (t^(-1/2) decay)"
                }
                if not passed:
                    summary["all_passed"] = False

        if "loss_scaling" in self.results:
            result = self.results["loss_scaling"]
            beta = get_metric(result, "scaling_exponent")
            if beta is not None:
                summary["exponents"]["loss_scaling_beta"] = beta
                passed = abs(beta - self.EXPECTED_BETA) <= self.TOLERANCE
                summary["predictions"]["loss_scaling"] = {
                    "expected": self.EXPECTED_BETA,
                    "observed": beta,
                    "passed": passed,
                    "description": f"beta ~ {self.EXPECTED_BETA} (c^(-0.5) scaling)"
                }
                if not passed:
                    summary["all_passed"] = False

        if "mem0_retrieval" in self.results:
            result = self.results["mem0_retrieval"]
            beta = get_metric(result, "retrieval_decay_exponent") or get_metric(result, "decay_exponent")
            if beta is not None:
                summary["exponents"]["mem0_retrieval_beta"] = beta
                passed = abs(beta - self.EXPECTED_BETA) <= self.TOLERANCE
                summary["predictions"]["mem0_retrieval"] = {
                    "expected": self.EXPECTED_BETA,
                    "observed": beta,
                    "passed": passed,
                    "description": f"beta ~ {self.EXPECTED_BETA} (retrieval decay)"
                }
                if not passed:
                    summary["all_passed"] = False

        return summary

    def _generate_plots(self) -> None:
        """Generate plots for all experiment results."""
        experiments_data = []

        # Helper to extract data from result
        def extract_plot_data(result, name, x_key, y_key, x_label, y_label, exp_key):
            if result is None:
                return None
            data = result.data if hasattr(result, 'data') else result.get('data', {})
            metrics = result.metrics if hasattr(result, 'metrics') else result.get('metrics', {})

            x_data = data.get(x_key, [])
            y_data = data.get(y_key, [])
            exponent = metrics.get(exp_key, 0)
            r_squared = metrics.get('r_squared', 0)

            if len(x_data) == 0 or len(y_data) == 0:
                return None

            return {
                "name": name,
                "x_data": x_data,
                "y_data": y_data,
                "exponent": exponent,
                "r_squared": r_squared,
                "x_label": x_label,
                "y_label": y_label,
                "expected_exponent": 0.5
            }

        # Extract data from each experiment
        if "embedding_drift" in self.results:
            plot_data = extract_plot_data(
                self.results["embedding_drift"],
                "Embedding Drift (H~0.5)",
                "positions", "variances",
                "Token Position", "Variance σ²(t)",
                "hurst_exponent"
            )
            if plot_data:
                experiments_data.append(plot_data)

        if "alignment_decay" in self.results:
            plot_data = extract_plot_data(
                self.results["alignment_decay"],
                "Alignment Decay (β~0.5)",
                "positions", "alignments",
                "Context Length", "Alignment",
                "decay_exponent"
            )
            if plot_data:
                experiments_data.append(plot_data)

        if "loss_scaling" in self.results:
            plot_data = extract_plot_data(
                self.results["loss_scaling"],
                "Loss Scaling (β~0.5)",
                "context_lengths", "losses",
                "Context Length", "Loss",
                "scaling_exponent"
            )
            if plot_data:
                experiments_data.append(plot_data)

        if "mem0_retrieval" in self.results:
            plot_data = extract_plot_data(
                self.results["mem0_retrieval"],
                "Memory Retrieval Decay",
                "memory_ages", "retrieval_scores",
                "Memory Age", "Retrieval Score",
                "decay_exponent"
            )
            if plot_data:
                experiments_data.append(plot_data)

        if experiments_data:
            self.visualizer.plot_all_experiments(experiments_data, "Query Drift Hypothesis Validation")

    def _save_results(self) -> None:
        """Save results to JSON file."""
        output_dir = self.config.output.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "query_drift_results.json")

        # Prepare serializable results
        output_data = {
            "config": self.config.to_dict(),
            "results": self.results,
            "summary": self.summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        if self.config.output.verbose:
            print(f"\nResults saved to: {output_path}")

    def _print_header(self) -> None:
        """Print experiment header."""
        print("\n" + "=" * 70)
        print("  QUERY DRIFT HYPOTHESIS VALIDATION")
        print("=" * 70)
        print("\nTesting predictions:")
        print("  - Embedding drift follows Brownian motion (H ~ 0.5)")
        print("  - Alignment decays as t^(-1/2) with context length")
        print("  - Loss scales as c^(-0.5) with context length")
        print("  - Memory retrieval degrades following power law decay")
        print("-" * 70)

    def _print_experiment_start(self, name: str) -> None:
        """Print experiment start message."""
        if self.config.output.verbose:
            print(f"\n>>> Running {name} Experiment...")

    def _print_experiment_complete(self, name: str) -> None:
        """Print experiment completion message."""
        if self.config.output.verbose:
            print(f"    {name} Experiment complete.")

    def _print_summary(self) -> None:
        """Print formatted summary table of results."""
        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)

        # Print table header
        print(f"\n{'Experiment':<25} {'Expected':<12} {'Observed':<12} {'Status':<10}")
        print("-" * 60)

        # Print results for each experiment
        for exp_name, prediction in self.summary.get("predictions", {}).items():
            expected = f"{prediction['expected']:.3f}"
            observed = f"{prediction['observed']:.3f}"
            status = "PASS" if prediction["passed"] else "FAIL"
            status_marker = "[+]" if prediction["passed"] else "[-]"

            # Format experiment name for display
            display_name = exp_name.replace("_", " ").title()

            print(f"{display_name:<25} {expected:<12} {observed:<12} {status_marker} {status}")

        print("-" * 60)

        # Print overall result
        all_passed = self.summary.get("all_passed", False)
        if all_passed:
            print("\n[+] ALL PREDICTIONS PASSED")
            print("    The Query Drift Hypothesis is SUPPORTED by these experiments.")
        else:
            passed_count = sum(
                1 for p in self.summary.get("predictions", {}).values()
                if p.get("passed", False)
            )
            total_count = len(self.summary.get("predictions", {}))
            print(f"\n[-] {passed_count}/{total_count} PREDICTIONS PASSED")
            print("    Some predictions did not match expected values.")
            print("    Consider increasing sample sizes or investigating deviations.")

        # Print interpretation
        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        print("-" * 70)
        print("""
The Query Drift Hypothesis predicts that attention mechanisms in
transformers exhibit Brownian-like drift in embedding space over the
course of generation. This manifests as:

1. EMBEDDING DRIFT (H ~ 0.5): The Hurst exponent of ~0.5 indicates
   the embedding trajectory follows standard Brownian motion, neither
   trending nor mean-reverting.

2. ALIGNMENT DECAY (beta ~ 0.5): Attention alignment to original
   queries decays as t^(-1/2), consistent with random walk dispersion.

3. LOSS SCALING (beta ~ 0.5): Per-token loss contribution scales as
   c^(-0.5) with context length, matching diffusive information loss.

4. MEMORY RETRIEVAL (beta ~ 0.5): External memory retrieval quality
   degrades following the same power law, suggesting the drift affects
   similarity-based retrieval systems.
""")

        # Print timing
        if "total_time_seconds" in self.summary:
            print(f"\nTotal execution time: {self.summary['total_time_seconds']:.2f} seconds")

        print("=" * 70 + "\n")
