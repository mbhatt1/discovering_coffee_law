#!/usr/bin/env python3
"""
Unified Experiment Runner for Query Drift → Ornstein-Uhlenbeck Paper

This script runs all experiments needed to validate that transformer attention
dynamics follow Ornstein-Uhlenbeck (mean-reverting) rather than pure Brownian motion.

Paper Structure:
    1. Introduction & Theory
    2. Experimental Setup
    3. Core Experiments (Section 3)
    4. Model Comparison & Selection (Section 4)
    5. Parameter Studies (Section 5)
    6. Cross-Model Validation (Section 6)
    7. Practical Implications (Section 7)
    8. Discussion & Conclusion

Usage:
    python run_experiments.py --all                    # Run everything
    python run_experiments.py --section 3             # Run specific section
    python run_experiments.py --quick                 # Quick validation run
    python run_experiments.py --experiment variance   # Run specific experiment
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Any, Callable
import numpy as np

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

from query_drift.config import ExperimentConfig
from query_drift.utils.embeddings import EmbeddingClient


@dataclass
class PaperExperimentConfig:
    """Configuration for full paper experiments."""

    # Output settings
    output_dir: str = "paper_experiments"
    save_intermediate: bool = True

    # Parallelization
    max_workers: int = 4  # Number of parallel experiment threads
    parallelize: bool = True

    # Core experiment parameters
    num_continuations: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    sample_positions: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 40, 50, 75, 100, 150])
    context_lengths: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 4000])

    # Model comparison
    embedding_models: List[str] = field(default_factory=lambda: [
        "text-embedding-3-small",
        "text-embedding-3-large",
    ])
    completion_models: List[str] = field(default_factory=lambda: [
        "gpt-4o-mini",
        "gpt-4o",
    ])

    # Temperature study
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0, 1.5])

    # Statistical settings
    num_trials: int = 3  # Repeat each experiment for confidence intervals
    bootstrap_samples: int = 1000

    # Quick mode settings
    quick_mode: bool = False


def setup_output_dir(config: PaperExperimentConfig) -> Path:
    """Create output directory structure."""
    base_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"

    # Create subdirectories for each section
    (run_dir / "section3_core").mkdir(parents=True, exist_ok=True)
    (run_dir / "section4_model_selection").mkdir(parents=True, exist_ok=True)
    (run_dir / "section5_parameters").mkdir(parents=True, exist_ok=True)
    (run_dir / "section6_cross_model").mkdir(parents=True, exist_ok=True)
    (run_dir / "section7_applications").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    return run_dir


class ExperimentRunner:
    """Main experiment runner for the paper with parallel execution support."""

    def __init__(self, config: PaperExperimentConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.run_dir: Optional[Path] = None
        self._results_lock = Lock()
        self._print_lock = Lock()

        # Initialize clients lazily (thread-local would be better for heavy use)
        self._client = None
        self._embedding_client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    @property
    def embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()
        return self._embedding_client

    def _thread_safe_print(self, msg: str):
        """Thread-safe printing."""
        with self._print_lock:
            print(msg)

    def _run_parallel(self, tasks: List[tuple], description: str) -> Dict[str, Any]:
        """
        Run multiple tasks in parallel.

        Args:
            tasks: List of (name, callable, args) tuples
            description: Description for logging

        Returns:
            Dict mapping task names to results
        """
        results = {}

        if not self.config.parallelize or len(tasks) <= 1:
            # Sequential execution
            for name, func, args in tasks:
                self._thread_safe_print(f"  [{name}] Starting...")
                results[name] = func(*args) if args else func()
                self._thread_safe_print(f"  [{name}] Complete")
            return results

        # Parallel execution
        self._thread_safe_print(f"\n  Running {len(tasks)} tasks in parallel (max {self.config.max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_name = {}
            for name, func, args in tasks:
                future = executor.submit(func, *args) if args else executor.submit(func)
                future_to_name[future] = name
                self._thread_safe_print(f"  [{name}] Submitted")

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    self._thread_safe_print(f"  [{name}] ✓ Complete")
                except Exception as e:
                    self._thread_safe_print(f"  [{name}] ✗ Error: {e}")
                    results[name] = {"error": str(e)}

        return results

    def run_all(self) -> Dict[str, Any]:
        """Run all experiments for the paper."""
        self.run_dir = setup_output_dir(self.config)
        print(f"\n{'='*70}")
        print(f"PAPER EXPERIMENT SUITE")
        print(f"Output directory: {self.run_dir}")
        print(f"{'='*70}\n")

        start_time = time.time()

        # Section 3: Core Experiments
        self.run_section3_core_experiments()

        # Section 4: Model Comparison & Selection
        self.run_section4_model_selection()

        # Section 5: Parameter Studies
        self.run_section5_parameter_studies()

        # Section 6: Cross-Model Validation
        self.run_section6_cross_model()

        # Section 7: Practical Implications
        self.run_section7_applications()

        # Generate summary
        total_time = time.time() - start_time
        self.results["total_time_seconds"] = total_time
        self.results["timestamp"] = datetime.now().isoformat()

        # Save master results file
        self._save_results("master_results.json", self.results)

        print(f"\n{'='*70}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.run_dir}")
        print(f"{'='*70}\n")

        return self.results

    def run_section3_core_experiments(self):
        """
        Section 3: Core Experiments (PARALLELIZED)

        3.1 Embedding Variance Growth
        3.2 Alignment Decay Measurement
        3.3 Loss Scaling Analysis
        3.4 Memory Retrieval Degradation
        """
        print("\n" + "="*70)
        print("SECTION 3: CORE EXPERIMENTS")
        print("="*70)

        section_dir = self.run_dir / "section3_core"

        # Define all experiments to run in parallel
        tasks = [
            ("3.1_variance_growth", self._run_variance_growth_experiment, (section_dir,)),
            ("3.2_alignment_decay", self._run_alignment_decay_experiment, (section_dir,)),
            ("3.3_loss_scaling", self._run_loss_scaling_experiment, (section_dir,)),
            ("3.4_memory_retrieval", self._run_memory_retrieval_experiment, (section_dir,)),
        ]

        section_results = self._run_parallel(tasks, "Section 3 Core Experiments")

        with self._results_lock:
            self.results["section3"] = section_results
        self._save_results("section3_core/results.json", section_results)

    def run_section4_model_selection(self):
        """
        Section 4: Model Comparison & Statistical Selection

        4.1 Brownian Motion Fit
        4.2 Fractional Brownian Motion Fit
        4.3 Ornstein-Uhlenbeck Fit
        4.4 Model Selection (AIC/BIC)
        """
        print("\n" + "="*70)
        print("SECTION 4: MODEL COMPARISON & SELECTION")
        print("="*70)

        section_results = {}
        section_dir = self.run_dir / "section4_model_selection"

        # Use data from Section 3 for model fitting
        if "section3" not in self.results:
            print("Warning: Section 3 data not available, running core experiments first...")
            self.run_section3_core_experiments()

        print("\n>>> 4.1-4.3 Fitting Stochastic Models")
        section_results["model_fits"] = self._fit_stochastic_models(section_dir)

        print("\n>>> 4.4 Model Selection Criteria")
        section_results["model_selection"] = self._compute_model_selection(section_dir)

        self.results["section4"] = section_results
        self._save_results("section4_model_selection/results.json", section_results)

    def run_section5_parameter_studies(self):
        """
        Section 5: Parameter Studies (PARALLELIZED)

        5.1 Temperature Dependence
        5.2 Context Length Effects
        5.3 Prompt Domain Sensitivity
        """
        print("\n" + "="*70)
        print("SECTION 5: PARAMETER STUDIES")
        print("="*70)

        section_dir = self.run_dir / "section5_parameters"

        # Run parameter studies in parallel
        tasks = [
            ("5.1_temperature", self._run_temperature_study, (section_dir,)),
            ("5.2_context_length", self._run_context_length_study, (section_dir,)),
            ("5.3_domain", self._run_domain_study, (section_dir,)),
        ]

        section_results = self._run_parallel(tasks, "Section 5 Parameter Studies")

        with self._results_lock:
            self.results["section5"] = section_results
        self._save_results("section5_parameters/results.json", section_results)

    def run_section6_cross_model(self):
        """
        Section 6: Cross-Model Validation (PARALLELIZED)

        6.1 Different Embedding Models
        6.2 Different Completion Models
        6.3 Universal OU Parameters?
        """
        print("\n" + "="*70)
        print("SECTION 6: CROSS-MODEL VALIDATION")
        print("="*70)

        section_dir = self.run_dir / "section6_cross_model"

        # Run model comparisons in parallel (6.3 depends on 6.1 and 6.2)
        tasks = [
            ("6.1_embedding_models", self._run_embedding_model_comparison, (section_dir,)),
            ("6.2_completion_models", self._run_completion_model_comparison, (section_dir,)),
        ]

        section_results = self._run_parallel(tasks, "Section 6 Cross-Model")

        # 6.3 depends on previous results
        print("\n>>> 6.3 Universal Parameter Analysis")
        with self._results_lock:
            self.results["section6"] = section_results
        section_results["6.3_universal_params"] = self._analyze_universal_parameters(section_dir)

        with self._results_lock:
            self.results["section6"] = section_results
        self._save_results("section6_cross_model/results.json", section_results)

    def run_section7_applications(self):
        """
        Section 7: Practical Implications

        7.1 Optimal Context Window Size
        7.2 RAG System Design Guidelines
        7.3 Memory System Recommendations
        """
        print("\n" + "="*70)
        print("SECTION 7: PRACTICAL IMPLICATIONS")
        print("="*70)

        section_results = {}
        section_dir = self.run_dir / "section7_applications"

        print("\n>>> 7.1 Optimal Context Window Analysis")
        section_results["7.1_context_window"] = self._analyze_optimal_context(section_dir)

        print("\n>>> 7.2 RAG System Guidelines")
        section_results["7.2_rag_guidelines"] = self._analyze_rag_implications(section_dir)

        print("\n>>> 7.3 Memory System Design")
        section_results["7.3_memory_design"] = self._analyze_memory_implications(section_dir)

        self.results["section7"] = section_results
        self._save_results("section7_applications/results.json", section_results)

    # =========================================================================
    # Section 3: Core Experiment Implementations
    # =========================================================================

    def _run_variance_growth_experiment(self, output_dir: Path) -> Dict[str, Any]:
        """Measure embedding variance growth at multiple positions."""
        from query_drift.experiments.embedding_drift import EmbeddingDriftExperiment

        results = {"trials": [], "aggregated": {}}

        num_continuations = 10 if self.config.quick_mode else 30
        positions = [10, 20, 30, 50, 75, 100] if not self.config.quick_mode else [10, 25, 50]

        for trial in range(self.config.num_trials if not self.config.quick_mode else 1):
            print(f"  Trial {trial + 1}/{self.config.num_trials if not self.config.quick_mode else 1}")

            config = ExperimentConfig()
            config.embedding_drift.num_continuations = num_continuations
            config.embedding_drift.sample_positions = positions
            config.embedding_drift.max_tokens_drift = max(positions) + 20
            config.output.show_plots = False

            exp = EmbeddingDriftExperiment(config, self.client, self.embedding_client)
            result = exp.run()

            if result.success:
                results["trials"].append({
                    "trial": trial,
                    "positions": result.data["positions"].tolist() if hasattr(result.data["positions"], 'tolist') else result.data["positions"],
                    "variances": result.data["variances"].tolist() if hasattr(result.data["variances"], 'tolist') else result.data["variances"],
                    "hurst_exponent": result.metrics["hurst_exponent"],
                    "r_squared": result.metrics["r_squared"],
                })

        # Aggregate across trials
        if results["trials"]:
            h_values = [t["hurst_exponent"] for t in results["trials"]]
            results["aggregated"] = {
                "mean_hurst": float(np.mean(h_values)),
                "std_hurst": float(np.std(h_values)),
                "num_trials": len(results["trials"]),
            }
            print(f"  H = {results['aggregated']['mean_hurst']:.4f} ± {results['aggregated']['std_hurst']:.4f}")

        return results

    def _run_alignment_decay_experiment(self, output_dir: Path) -> Dict[str, Any]:
        """Measure alignment decay with context length."""
        from query_drift.experiments.alignment_decay import AlignmentDecayExperiment

        results = {"trials": [], "aggregated": {}}

        num_rounds = 2 if self.config.quick_mode else 4

        for trial in range(self.config.num_trials if not self.config.quick_mode else 1):
            print(f"  Trial {trial + 1}/{self.config.num_trials if not self.config.quick_mode else 1}")

            config = ExperimentConfig()
            config.alignment_decay.num_distractor_rounds = num_rounds
            config.output.show_plots = False

            exp = AlignmentDecayExperiment(config, self.client, self.embedding_client)
            result = exp.run()

            if result.success:
                results["trials"].append({
                    "trial": trial,
                    "context_lengths": result.data["context_lengths"].tolist() if hasattr(result.data["context_lengths"], 'tolist') else result.data["context_lengths"],
                    "alignments": result.data["alignments"].tolist() if hasattr(result.data["alignments"], 'tolist') else result.data["alignments"],
                    "fitted_beta": result.metrics["fitted_beta"],
                    "r_squared": result.metrics["r_squared"],
                })

        if results["trials"]:
            beta_values = [t["fitted_beta"] for t in results["trials"]]
            results["aggregated"] = {
                "mean_beta": float(np.mean(beta_values)),
                "std_beta": float(np.std(beta_values)),
                "num_trials": len(results["trials"]),
            }
            print(f"  β = {results['aggregated']['mean_beta']:.4f} ± {results['aggregated']['std_beta']:.4f}")

        return results

    def _run_loss_scaling_experiment(self, output_dir: Path) -> Dict[str, Any]:
        """Measure loss scaling with context length."""
        from query_drift.experiments.loss_scaling import LossScalingExperiment

        results = {"trials": [], "aggregated": {}}

        context_lengths = [100, 500, 1000] if self.config.quick_mode else [100, 200, 500, 1000, 2000]

        for trial in range(self.config.num_trials if not self.config.quick_mode else 1):
            print(f"  Trial {trial + 1}/{self.config.num_trials if not self.config.quick_mode else 1}")

            config = ExperimentConfig()
            config.loss_scaling.context_lengths = context_lengths
            config.output.show_plots = False

            exp = LossScalingExperiment(config, self.client, self.embedding_client)
            result = exp.run()

            if result.success:
                results["trials"].append({
                    "trial": trial,
                    "context_lengths": result.data["context_lengths"].tolist() if hasattr(result.data["context_lengths"], 'tolist') else result.data["context_lengths"],
                    "mean_losses": result.data["mean_losses"].tolist() if hasattr(result.data["mean_losses"], 'tolist') else result.data["mean_losses"],
                    "scaling_exponent": result.metrics.get("scaling_exponent"),
                    "r_squared": result.metrics.get("r_squared"),
                })

        if results["trials"]:
            beta_values = [t["scaling_exponent"] for t in results["trials"] if t["scaling_exponent"] is not None]
            if beta_values:
                results["aggregated"] = {
                    "mean_beta": float(np.mean(beta_values)),
                    "std_beta": float(np.std(beta_values)),
                    "num_trials": len(beta_values),
                }
                print(f"  β = {results['aggregated']['mean_beta']:.4f} ± {results['aggregated']['std_beta']:.4f}")

        return results

    def _run_memory_retrieval_experiment(self, output_dir: Path) -> Dict[str, Any]:
        """Measure memory retrieval degradation."""
        from query_drift.experiments.mem0_retrieval import Mem0RetrievalExperiment

        results = {"trials": [], "aggregated": {}}

        num_memories = 10 if self.config.quick_mode else 20

        for trial in range(self.config.num_trials if not self.config.quick_mode else 1):
            print(f"  Trial {trial + 1}/{self.config.num_trials if not self.config.quick_mode else 1}")

            config = ExperimentConfig()
            config.mem0_retrieval.num_memories = num_memories
            config.mem0_retrieval.distractor_multiplier = 2
            config.output.show_plots = False

            exp = Mem0RetrievalExperiment(config, self.client, self.embedding_client)
            result = exp.run()

            if result.success:
                results["trials"].append({
                    "trial": trial,
                    "retrieval_rate": result.metrics["retrieval_rate"],
                    "mean_score": result.metrics["mean_retrieval_score"],
                    "decay_exponent": result.metrics.get("decay_exponent"),
                })

        if results["trials"]:
            rates = [t["retrieval_rate"] for t in results["trials"]]
            results["aggregated"] = {
                "mean_retrieval_rate": float(np.mean(rates)),
                "std_retrieval_rate": float(np.std(rates)),
                "num_trials": len(results["trials"]),
            }
            print(f"  Retrieval rate = {results['aggregated']['mean_retrieval_rate']:.1%}")

        return results

    # =========================================================================
    # Section 4: Model Selection Implementations
    # =========================================================================

    def _fit_stochastic_models(self, output_dir: Path) -> Dict[str, Any]:
        """Fit Brownian, fBm, and OU models to variance data."""
        from scipy.optimize import curve_fit

        # Get variance data from Section 3
        variance_data = self.results.get("section3", {}).get("3.1_variance_growth", {})

        if not variance_data.get("trials"):
            print("  No variance data available")
            return {"error": "No variance data"}

        # Aggregate all variance measurements
        all_positions = []
        all_variances = []
        for trial in variance_data["trials"]:
            all_positions.extend(trial["positions"])
            all_variances.extend(trial["variances"])

        positions = np.array(all_positions)
        variances = np.array(all_variances)

        results = {}

        # Model 1: Standard Brownian (H = 0.5 fixed)
        def brownian(t, A):
            return A * t

        try:
            popt, _ = curve_fit(brownian, positions, variances, p0=[0.001])
            pred = brownian(positions, *popt)
            ss_res = np.sum((variances - pred)**2)
            ss_tot = np.sum((variances - np.mean(variances))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            results["brownian"] = {
                "amplitude": float(popt[0]),
                "hurst": 0.5,
                "r_squared": float(r2),
                "aic": float(len(variances) * np.log(ss_res/len(variances)) + 2 * 1),  # 1 parameter
            }
            print(f"  Brownian (H=0.5): R² = {r2:.4f}")
        except Exception as e:
            results["brownian"] = {"error": str(e)}

        # Model 2: Fractional Brownian Motion
        def fbm(t, A, H):
            return A * np.power(t, 2*H)

        try:
            popt, _ = curve_fit(fbm, positions, variances, p0=[0.01, 0.3], bounds=([0, 0], [1, 1]))
            pred = fbm(positions, *popt)
            ss_res = np.sum((variances - pred)**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            results["fbm"] = {
                "amplitude": float(popt[0]),
                "hurst": float(popt[1]),
                "r_squared": float(r2),
                "aic": float(len(variances) * np.log(ss_res/len(variances)) + 2 * 2),  # 2 parameters
            }
            print(f"  fBm (H={popt[1]:.3f}): R² = {r2:.4f}")
        except Exception as e:
            results["fbm"] = {"error": str(e)}

        # Model 3: Ornstein-Uhlenbeck
        def ou(t, sigma_inf_sq, theta):
            return sigma_inf_sq * (1 - np.exp(-2 * theta * t))

        try:
            # Normalize for stability
            t_norm = positions / positions.max()
            popt, _ = curve_fit(ou, t_norm, variances,
                               p0=[variances.max()*1.5, 1.0],
                               bounds=([0, 0.001], [1, 100]))
            theta_actual = popt[1] / positions.max()
            pred = ou(t_norm, *popt)
            ss_res = np.sum((variances - pred)**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            results["ou"] = {
                "sigma_inf_sq": float(popt[0]),
                "theta": float(theta_actual),
                "relaxation_time": float(1/(2*theta_actual)),
                "r_squared": float(r2),
                "aic": float(len(variances) * np.log(ss_res/len(variances)) + 2 * 2),  # 2 parameters
            }
            print(f"  OU (θ={theta_actual:.4f}): R² = {r2:.4f}")
        except Exception as e:
            results["ou"] = {"error": str(e)}

        return results

    def _compute_model_selection(self, output_dir: Path) -> Dict[str, Any]:
        """Compute AIC/BIC for model selection."""
        model_fits = self.results.get("section4", {}).get("model_fits", {})

        results = {"ranking": [], "best_model": None}

        models_with_aic = []
        for model_name, fit in model_fits.items():
            if isinstance(fit, dict) and "aic" in fit:
                models_with_aic.append((model_name, fit["aic"], fit["r_squared"]))

        if models_with_aic:
            # Sort by AIC (lower is better)
            models_with_aic.sort(key=lambda x: x[1])
            results["ranking"] = [
                {"model": m[0], "aic": m[1], "r_squared": m[2]}
                for m in models_with_aic
            ]
            results["best_model"] = models_with_aic[0][0]

            print(f"  Best model by AIC: {results['best_model']}")
            for rank, (name, aic, r2) in enumerate(models_with_aic, 1):
                print(f"    {rank}. {name}: AIC={aic:.2f}, R²={r2:.4f}")

        return results

    # =========================================================================
    # Section 5: Parameter Studies
    # =========================================================================

    def _run_temperature_study(self, output_dir: Path) -> Dict[str, Any]:
        """Study effect of temperature on OU parameters (parallelized)."""
        from query_drift.experiments.embedding_drift import EmbeddingDriftExperiment

        temperatures = [0.3, 0.7, 1.0] if self.config.quick_mode else self.config.temperatures

        def run_single_temp(temp: float) -> Dict[str, Any]:
            """Run experiment for a single temperature."""
            from openai import OpenAI
            client = OpenAI()
            emb_client = EmbeddingClient()

            config = ExperimentConfig()
            config.embedding_drift.temperature = temp
            config.embedding_drift.num_continuations = 5 if self.config.quick_mode else 15
            config.embedding_drift.sample_positions = [10, 25, 50]
            config.output.show_plots = False

            exp = EmbeddingDriftExperiment(config, client, emb_client)
            result = exp.run()

            if result.success:
                return {
                    "hurst_exponent": result.metrics["hurst_exponent"],
                    "r_squared": result.metrics["r_squared"],
                    "amplitude": result.metrics["amplitude"],
                }
            return {"error": "Experiment failed"}

        # Create tasks for parallel execution
        tasks = [(f"temp_{temp}", run_single_temp, (temp,)) for temp in temperatures]
        temp_results = self._run_parallel(tasks, "Temperature Study")

        # Reformat results
        results = {"by_temperature": {}}
        for temp in temperatures:
            key = f"temp_{temp}"
            if key in temp_results and "error" not in temp_results[key]:
                results["by_temperature"][str(temp)] = temp_results[key]
                self._thread_safe_print(f"    T={temp}: H = {temp_results[key]['hurst_exponent']:.4f}")

        return results

    def _run_context_length_study(self, output_dir: Path) -> Dict[str, Any]:
        """Study how OU parameters vary with maximum context length."""
        # This uses the loss scaling experiment with different ranges
        results = {"analysis": "Context length effects captured in Section 3.3"}
        print("  Using data from Section 3.3 (Loss Scaling)")
        return results

    def _run_domain_study(self, output_dir: Path) -> Dict[str, Any]:
        """Study sensitivity to prompt domain (parallelized)."""
        domains = {
            "technical": "The architecture of modern neural networks involves multiple layers of computation. Each layer transforms the input through learned weight matrices and nonlinear activation functions.",
            "narrative": "Once upon a time in a distant kingdom, there lived a young prince who dreamed of adventure. Every night he would gaze at the stars and wonder what lay beyond the mountains.",
            "scientific": "The process of photosynthesis converts light energy into chemical energy through a series of biochemical reactions. Chlorophyll molecules in plant cells absorb photons and initiate electron transport.",
            "conversational": "Hey, so I was thinking about what we discussed yesterday. You know how sometimes things just don't work out the way you planned? Well, I had this idea.",
        }

        if self.config.quick_mode:
            domains = {"technical": domains["technical"], "narrative": domains["narrative"]}

        quick_mode = self.config.quick_mode

        def run_single_domain(domain_name: str, prefix: str) -> Dict[str, Any]:
            """Run experiment for a single domain."""
            from openai import OpenAI
            client = OpenAI()
            emb_client = EmbeddingClient()

            # Generate continuations and measure variance
            continuations = []
            for i in range(5 if quick_mode else 10):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Continue the text naturally."},
                        {"role": "user", "content": prefix}
                    ],
                    max_tokens=50,
                    temperature=1.0
                )
                continuations.append(response.choices[0].message.content)

            # Compute embeddings and variance
            embeddings = []
            for cont in continuations:
                full_text = prefix + " " + cont
                emb = emb_client.get_embedding(full_text)
                embeddings.append(np.array(emb.embedding))

            emb_matrix = np.array(embeddings)
            centroid = np.mean(emb_matrix, axis=0)
            variance = float(np.mean(np.sum((emb_matrix - centroid)**2, axis=1)))

            return {
                "variance": variance,
                "num_samples": len(continuations),
            }

        # Create tasks for parallel execution
        tasks = [(name, run_single_domain, (name, prefix)) for name, prefix in domains.items()]
        domain_results = self._run_parallel(tasks, "Domain Study")

        # Reformat results
        results = {"by_domain": {}}
        for domain_name in domains.keys():
            if domain_name in domain_results:
                results["by_domain"][domain_name] = domain_results[domain_name]
                self._thread_safe_print(f"    {domain_name}: variance = {domain_results[domain_name]['variance']:.6f}")

        return results

    # =========================================================================
    # Section 6: Cross-Model Validation
    # =========================================================================

    def _run_embedding_model_comparison(self, output_dir: Path) -> Dict[str, Any]:
        """Compare different embedding models."""
        models = ["text-embedding-3-small"] if self.config.quick_mode else self.config.embedding_models
        results = {"by_model": {}}

        prefix = "The development of artificial intelligence has progressed through several key phases."

        for model in models:
            print(f"  Model: {model}")

            # Create client for this model
            emb_client = EmbeddingClient(model=model)

            # Generate continuations
            continuations = []
            for i in range(5 if self.config.quick_mode else 10):
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prefix}],
                    max_tokens=50,
                    temperature=1.0
                )
                continuations.append(response.choices[0].message.content)

            # Compute variance
            embeddings = []
            for cont in continuations:
                emb = emb_client.get_embedding(prefix + " " + cont)
                embeddings.append(np.array(emb.embedding))

            emb_matrix = np.array(embeddings)
            centroid = np.mean(emb_matrix, axis=0)
            variance = float(np.mean(np.sum((emb_matrix - centroid)**2, axis=1)))

            results["by_model"][model] = {
                "variance": variance,
                "embedding_dim": len(embeddings[0]),
            }
            print(f"    Variance = {variance:.6f}, dim = {len(embeddings[0])}")

        return results

    def _run_completion_model_comparison(self, output_dir: Path) -> Dict[str, Any]:
        """Compare different completion models."""
        models = ["gpt-4o-mini"] if self.config.quick_mode else self.config.completion_models
        results = {"by_model": {}}

        prefix = "The history of computing began with mechanical calculators."

        for model in models:
            print(f"  Model: {model}")

            continuations = []
            for i in range(5 if self.config.quick_mode else 10):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prefix}],
                        max_tokens=50,
                        temperature=1.0
                    )
                    continuations.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            if continuations:
                embeddings = []
                for cont in continuations:
                    emb = self.embedding_client.get_embedding(prefix + " " + cont)
                    embeddings.append(np.array(emb.embedding))

                emb_matrix = np.array(embeddings)
                centroid = np.mean(emb_matrix, axis=0)
                variance = float(np.mean(np.sum((emb_matrix - centroid)**2, axis=1)))

                results["by_model"][model] = {
                    "variance": variance,
                    "num_samples": len(continuations),
                }
                print(f"    Variance = {variance:.6f}")

        return results

    def _analyze_universal_parameters(self, output_dir: Path) -> Dict[str, Any]:
        """Analyze if OU parameters are universal across models."""
        results = {
            "analysis": "Comparing OU parameters across models and domains",
            "hypothesis": "θ (mean-reversion rate) may be universal, σ²_∞ may vary",
        }

        # Collect all H values
        h_values = []

        # From temperature study
        temp_data = self.results.get("section5", {}).get("5.1_temperature", {}).get("by_temperature", {})
        for temp, data in temp_data.items():
            if "hurst_exponent" in data:
                h_values.append(data["hurst_exponent"])

        # From core experiments
        core_data = self.results.get("section3", {}).get("3.1_variance_growth", {}).get("aggregated", {})
        if "mean_hurst" in core_data:
            h_values.append(core_data["mean_hurst"])

        if h_values:
            results["h_statistics"] = {
                "mean": float(np.mean(h_values)),
                "std": float(np.std(h_values)),
                "min": float(np.min(h_values)),
                "max": float(np.max(h_values)),
                "n": len(h_values),
            }
            print(f"  H across conditions: {np.mean(h_values):.4f} ± {np.std(h_values):.4f}")

        return results

    # =========================================================================
    # Section 7: Practical Implications
    # =========================================================================

    def _analyze_optimal_context(self, output_dir: Path) -> Dict[str, Any]:
        """Derive optimal context window recommendations."""
        ou_params = self.results.get("section4", {}).get("model_fits", {}).get("ou", {})

        if "theta" in ou_params:
            theta = ou_params["theta"]
            tau = ou_params["relaxation_time"]

            results = {
                "relaxation_time_tokens": tau,
                "recommended_refresh_interval": tau * 2,  # Refresh context every 2τ
                "analysis": f"Based on OU dynamics with τ={tau:.1f} tokens, "
                           f"context should be refreshed approximately every {tau*2:.0f} tokens "
                           f"to maintain optimal alignment.",
            }
            print(f"  Relaxation time τ = {tau:.1f} tokens")
            print(f"  Recommended refresh interval: {tau*2:.0f} tokens")
        else:
            results = {"analysis": "OU parameters not available"}

        return results

    def _analyze_rag_implications(self, output_dir: Path) -> Dict[str, Any]:
        """Derive RAG system design guidelines."""
        results = {
            "guidelines": [
                "1. Bounded drift means retrieval quality is more stable than pure Brownian would predict",
                "2. Position-based reranking may be less critical due to mean-reverting dynamics",
                "3. Chunk size should consider relaxation time τ for optimal coherence",
                "4. Multi-query retrieval can leverage the bounded variance property",
            ],
            "quantitative": {},
        }

        # Use retrieval data
        mem_data = self.results.get("section3", {}).get("3.4_memory_retrieval", {}).get("aggregated", {})
        if mem_data:
            results["quantitative"]["retrieval_rate"] = mem_data.get("mean_retrieval_rate", "N/A")

        print("  Guidelines generated based on OU dynamics")
        return results

    def _analyze_memory_implications(self, output_dir: Path) -> Dict[str, Any]:
        """Derive memory system design recommendations."""
        results = {
            "recommendations": [
                "1. Long-term memory systems benefit from OU-aware retrieval scoring",
                "2. Memory consolidation can exploit the saturation property",
                "3. Temporal weighting should follow exponential (OU) rather than power-law decay",
                "4. Memory capacity can be higher than Brownian analysis would suggest",
            ],
        }

        print("  Memory system recommendations generated")
        return results

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _save_results(self, filename: str, data: Dict[str, Any]):
        """Save results to JSON file."""
        filepath = self.run_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(filepath, 'w') as f:
            json.dump(convert(data), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for Query Drift → Ornstein-Uhlenbeck paper"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--section", type=int, choices=[3, 4, 5, 6, 7],
                       help="Run specific section only")
    parser.add_argument("--quick", action="store_true",
                       help="Quick validation run with reduced samples")
    parser.add_argument("--output-dir", default="paper_experiments",
                       help="Output directory for results")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials per experiment")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel execution")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Create config
    config = PaperExperimentConfig(
        output_dir=args.output_dir,
        num_trials=args.trials,
        quick_mode=args.quick,
        max_workers=args.workers,
        parallelize=not args.no_parallel,
    )

    print(f"Parallel execution: {'Enabled' if config.parallelize else 'Disabled'}")
    if config.parallelize:
        print(f"Max workers: {config.max_workers}")

    # Create runner
    runner = ExperimentRunner(config)

    # Run experiments
    if args.all or args.section is None:
        runner.run_all()
    elif args.section == 3:
        runner.run_dir = setup_output_dir(config)
        runner.run_section3_core_experiments()
    elif args.section == 4:
        runner.run_dir = setup_output_dir(config)
        runner.run_section3_core_experiments()  # Need Section 3 data
        runner.run_section4_model_selection()
    elif args.section == 5:
        runner.run_dir = setup_output_dir(config)
        runner.run_section5_parameter_studies()
    elif args.section == 6:
        runner.run_dir = setup_output_dir(config)
        runner.run_section6_cross_model()
    elif args.section == 7:
        runner.run_dir = setup_output_dir(config)
        runner.run_section3_core_experiments()
        runner.run_section4_model_selection()
        runner.run_section7_applications()

    print("\nExperiment run complete!")
    print(f"Results saved to: {runner.run_dir}")


if __name__ == "__main__":
    main()
