"""
Alignment Decay Experiment

Validates the Query Drift Hypothesis prediction that alignment between
query embeddings and task direction decays as t^(-1/2) with context length.

The theoretical prediction is:
    C_t = <q_t, u> / ||q_t|| ~ t^(-1/2)

where:
    - q_t is the query embedding at context length t
    - u is the normalized task direction embedding
    - C_t is the cosine similarity (alignment)
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.math import cosine_similarity, fit_power_law


class AlignmentDecayExperiment(BaseExperiment):
    """
    Experiment to validate alignment decay follows t^(-1/2) power law.

    This experiment:
    1. Establishes a task direction embedding from the original query
    2. Progressively adds distractor text to the context
    3. Measures how alignment with task direction decays
    4. Fits a power law and validates exponent is approximately -0.5
    """

    experiment_name = "alignment_decay"

    def run(self) -> ExperimentResult:
        """
        Execute the alignment decay experiment.

        Returns:
            ExperimentResult with decay metrics and fitted parameters
        """
        self.log("Starting alignment decay experiment...")

        try:
            # Configuration
            decay_config = self.config.alignment_decay
            distractors = decay_config.distractors
            num_rounds = decay_config.num_distractor_rounds

            # Define task query and intro text
            task_query = "What are the key principles of machine learning?"
            intro_text = " Let me explain the fundamental concepts. "

            # Step 1: Get task direction embedding u and normalize it
            self.log("Computing task direction embedding...")
            task_result = self.embedding_client.get_embedding(task_query)
            u = task_result.embedding
            u = u / np.linalg.norm(u)  # Normalize task direction

            # Step 2: Start with base context
            base_context = task_query + intro_text

            # Track context lengths and alignments
            context_lengths = []
            alignments = []
            contexts = []

            # Measure initial alignment
            initial_result = self.embedding_client.get_embedding(base_context)
            q_0 = initial_result.embedding
            initial_alignment = cosine_similarity(q_0, u)

            context_lengths.append(len(base_context))
            alignments.append(initial_alignment)
            contexts.append(base_context)

            self.log(f"Initial context length: {len(base_context)}, alignment: {initial_alignment:.4f}")

            # Step 3: Add distractor sentences progressively
            current_context = base_context

            for round_idx in range(num_rounds):
                for distractor in distractors:
                    # Add distractor to context
                    current_context = current_context + distractor

                    # Step 4: Compute embedding and cosine similarity
                    context_result = self.embedding_client.get_embedding(current_context)
                    q_t = context_result.embedding

                    # Compute alignment C_t = <q_t, u> / ||q_t||
                    alignment = cosine_similarity(q_t, u)

                    context_lengths.append(len(current_context))
                    alignments.append(alignment)
                    contexts.append(current_context)

                    self.log(f"Context length: {len(current_context)}, alignment: {alignment:.4f}")

            # Convert to numpy arrays
            context_lengths = np.array(context_lengths)
            alignments = np.array(alignments)

            # Step 5: Fit power law decay C(t) = a * t^(-beta)
            self.log("Fitting power law decay...")

            # Use relative context lengths (normalized to first measurement)
            t_relative = context_lengths / context_lengths[0]

            # Fit power law to alignment data
            fit_result = fit_power_law(t_relative, alignments, with_offset=False)

            fitted_amplitude = fit_result.amplitude
            fitted_beta = -fit_result.exponent  # Convert to positive for display
            r_squared = fit_result.r_squared

            self.log(f"Fitted power law: C(t) = {fitted_amplitude:.4f} * t^(-{fitted_beta:.4f})")
            self.log(f"R-squared: {r_squared:.4f}")

            # Step 6: Validate beta approximately equals 0.5
            theoretical_beta = 0.5
            beta_error = abs(fitted_beta - theoretical_beta)
            beta_relative_error = beta_error / theoretical_beta

            # Consider validation successful if beta is within 50% of theoretical value
            beta_valid = beta_relative_error < 0.5

            self.log(f"Theoretical beta: {theoretical_beta}, Fitted beta: {fitted_beta:.4f}")
            self.log(f"Beta error: {beta_error:.4f} ({beta_relative_error*100:.1f}% relative error)")
            self.log(f"Validation {'PASSED' if beta_valid else 'FAILED'}")

            # Store result
            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=True,
                metrics={
                    "fitted_amplitude": float(fitted_amplitude),
                    "fitted_beta": float(fitted_beta),
                    "theoretical_beta": theoretical_beta,
                    "beta_error": float(beta_error),
                    "beta_relative_error": float(beta_relative_error),
                    "r_squared": float(r_squared),
                    "beta_valid": beta_valid,
                    "initial_alignment": float(initial_alignment),
                    "final_alignment": float(alignments[-1]),
                    "alignment_decay_ratio": float(alignments[-1] / initial_alignment),
                    "num_measurements": len(alignments),
                },
                data={
                    "context_lengths": context_lengths,
                    "alignments": alignments,
                    "t_relative": t_relative,
                    "task_direction": u,
                }
            )

            return self.result

        except Exception as e:
            self.log(f"Experiment failed with error: {str(e)}")
            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=False,
                error=str(e)
            )
            return self.result

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Generate log-log plot of alignment vs context length.

        Shows:
        - Measured alignment data points
        - Fitted power law curve
        - Theoretical t^(-0.5) decay line

        Args:
            save_path: Optional path to save the plot
        """
        if self.result is None or not self.result.success:
            self.log("No successful results to plot")
            return

        # Extract data
        t_relative = self.result.data["t_relative"]
        alignments = self.result.data["alignments"]
        context_lengths = self.result.data["context_lengths"]

        fitted_amplitude = self.result.metrics["fitted_amplitude"]
        fitted_beta = self.result.metrics["fitted_beta"]
        r_squared = self.result.metrics["r_squared"]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot measured data on log-log scale
        ax.loglog(
            t_relative, alignments,
            'o', markersize=8, color='blue', alpha=0.7,
            label='Measured alignment'
        )

        # Generate smooth curve for fitted power law
        t_smooth = np.logspace(0, np.log10(t_relative.max()), 100)

        # Fitted curve: C(t) = a * t^(-beta)
        fitted_curve = fitted_amplitude * np.power(t_smooth, -fitted_beta)
        ax.loglog(
            t_smooth, fitted_curve,
            '-', linewidth=2, color='red',
            label=f'Fitted: $C(t) = {fitted_amplitude:.3f} \\cdot t^{{-{fitted_beta:.3f}}}$ ($R^2={r_squared:.3f}$)'
        )

        # Theoretical t^(-0.5) line (scaled to match initial alignment)
        theoretical_curve = alignments[0] * np.power(t_smooth, -0.5)
        ax.loglog(
            t_smooth, theoretical_curve,
            '--', linewidth=2, color='green', alpha=0.7,
            label=r'Theoretical: $C(t) \propto t^{-0.5}$'
        )

        # Labels and title
        ax.set_xlabel('Relative Context Length (t / t$_0$)', fontsize=12)
        ax.set_ylabel('Alignment $C_t = \\langle q_t, u \\rangle / ||q_t||$', fontsize=12)
        ax.set_title(
            'Alignment Decay: Query-Task Direction Similarity vs Context Length\n'
            f'(Validation of $C_t \\sim t^{{-1/2}}$ hypothesis)',
            fontsize=14
        )

        # Add grid
        ax.grid(True, alpha=0.3, which='both')

        # Legend
        ax.legend(loc='upper right', fontsize=10)

        # Add text box with key metrics
        textstr = (
            f'Fitted $\\beta$: {fitted_beta:.4f}\n'
            f'Theoretical $\\beta$: 0.5\n'
            f'Error: {abs(fitted_beta - 0.5):.4f}\n'
            f'Validation: {"PASS" if self.result.metrics["beta_valid"] else "FAIL"}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props
        )

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi,
                       format=self.config.output.plot_format)
            self.log(f"Plot saved to {save_path}")

        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
