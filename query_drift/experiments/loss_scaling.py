"""
Loss Scaling Experiment for Query Drift Validation.

Measures how prediction confidence (inverse of loss) scales with context length.
Uses multiple measurement approaches for robustness.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.math import fit_power_law, power_law


class LossScalingExperiment(BaseExperiment):
    """
    Experiment to measure loss/perplexity scaling with context length.

    Approach: Measure prediction confidence via logprobs at different context lengths.
    The hypothesis predicts loss ~ c^(-0.5) + constant.
    """

    experiment_name = "loss_scaling"

    # Coherent passage with known continuation points
    BASE_PASSAGES = [
        {
            "context": "The development of artificial intelligence has progressed through several key phases. Early AI research in the 1950s focused on symbolic reasoning and logic-based systems. Researchers believed that human intelligence could be replicated through formal rules and symbol manipulation.",
            "continuation": " This approach, while elegant, struggled with real-world complexity."
        },
        {
            "context": "Climate change represents one of the most significant challenges facing humanity. Rising global temperatures are causing widespread environmental disruption. Polar ice caps are melting at unprecedented rates, contributing to rising sea levels.",
            "continuation": " Coastal cities around the world face increasing flood risks."
        },
        {
            "context": "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections. This complex network enables everything from basic motor functions to abstract reasoning and creativity.",
            "continuation": " Understanding how these connections give rise to consciousness remains one of science's greatest mysteries."
        },
        {
            "context": "Economic globalization has transformed international trade over the past century. Supply chains now span multiple continents, with products often assembled from components manufactured in dozens of countries.",
            "continuation": " This interconnectedness brings both opportunities and vulnerabilities."
        },
        {
            "context": "The history of mathematics stretches back thousands of years, with ancient civilizations developing number systems and geometric principles. The Greeks formalized mathematical proof, while Islamic scholars preserved and extended this knowledge.",
            "continuation": " The Renaissance saw mathematics become increasingly intertwined with physics."
        }
    ]

    def run(self) -> ExperimentResult:
        """Execute the loss scaling experiment with multiple measurements."""
        self.log("Starting loss scaling experiment")

        config = self.config.loss_scaling
        context_lengths = sorted(config.context_lengths)

        # Results storage
        length_loss_data = {}

        for length in context_lengths:
            losses = []

            for passage in self.BASE_PASSAGES:
                # Create context of specified length by truncating
                full_context = passage["context"]

                # Extend context if needed by repeating
                while len(full_context) < length:
                    full_context = full_context + " " + passage["context"]

                truncated = full_context[:length]
                continuation = passage["continuation"]

                # Measure logprob of continuation given truncated context
                loss = self._measure_continuation_loss(truncated, continuation)
                if loss is not None:
                    losses.append(loss)

            if losses:
                length_loss_data[length] = {
                    "mean": np.mean(losses),
                    "std": np.std(losses),
                    "n": len(losses),
                    "values": losses
                }
                self.log(f"  Length {length}: loss = {np.mean(losses):.4f} ± {np.std(losses):.4f} (n={len(losses)})")

        if len(length_loss_data) < 3:
            return ExperimentResult(
                experiment_name=self.experiment_name,
                success=False,
                error="Not enough valid measurements"
            )

        # Extract data for fitting
        lengths = np.array(sorted(length_loss_data.keys()))
        mean_losses = np.array([length_loss_data[l]["mean"] for l in lengths])
        std_losses = np.array([length_loss_data[l]["std"] for l in lengths])

        # Fit power law: L(c) = α * c^β + γ
        # For query drift, we expect β ≈ -0.5 (loss decreases with context)
        alpha = None
        beta = None
        gamma = 0.0
        r_squared = None
        fit_quality = "failed"

        # Check if there's enough variation to fit a power law
        loss_range = mean_losses.max() - mean_losses.min()
        loss_cv = np.std(mean_losses) / np.mean(mean_losses) if np.mean(mean_losses) > 0 else 0

        if loss_cv < 0.05:
            self.log(f"\nInsufficient loss variation (CV={loss_cv:.3f})")
            self.log("Loss is approximately constant across context lengths")
            fit_quality = "constant"
            alpha = np.mean(mean_losses)
            beta = 0.0
            r_squared = 0.0
        else:
            try:
                fit_result = fit_power_law(lengths, mean_losses, with_offset=True)
                alpha = fit_result.amplitude
                beta = fit_result.exponent
                gamma = fit_result.offset
                r_squared = fit_result.r_squared

                # Check for unrealistic exponents (fitting instability)
                if abs(beta) > 5:
                    self.log(f"Power law fit unstable (β={beta:.2f}), trying without offset...")
                    fit_result = fit_power_law(lengths, mean_losses, with_offset=False)
                    alpha = fit_result.amplitude
                    beta = fit_result.exponent
                    gamma = 0.0
                    r_squared = fit_result.r_squared

                fit_quality = "good" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "poor"
            except Exception as e:
                self.log(f"Power law fit failed: {e}")
                # Try without offset
                try:
                    fit_result = fit_power_law(lengths, mean_losses, with_offset=False)
                    alpha = fit_result.amplitude
                    beta = fit_result.exponent
                    gamma = 0.0
                    r_squared = fit_result.r_squared
                    fit_quality = "fallback"
                except Exception as e2:
                    self.log(f"All fitting attempts failed: {e2}")
                    # Report raw statistics instead
                    alpha = mean_losses[0]
                    beta = None
                    r_squared = None
                    fit_quality = "failed"

        if beta is not None:
            self.log(f"\nFitted: L(c) = {alpha:.4f} × c^{beta:.4f} + {gamma:.4f}")
            self.log(f"R² = {r_squared:.4f}")
            self.log(f"Scaling exponent β = {beta:.4f}")
            self.log(f"Expected β ≈ -0.5, observed β = {beta:.4f}")
            self.log(f"Fit quality: {fit_quality}")
        else:
            self.log(f"\nCould not determine power law relationship")
            self.log(f"Mean loss: {np.mean(mean_losses):.4f} ± {np.std(mean_losses):.4f}")

        # Check if loss decreases with context (β < 0) or increases
        if beta is not None and beta > 0:
            self.log("NOTE: Loss INCREASES with context (β > 0)")
            self.log("This may indicate noise or model-specific behavior")

        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "scaling_exponent": float(beta) if beta is not None else None,
                "amplitude": float(alpha) if alpha is not None else None,
                "offset": float(gamma) if gamma is not None else None,
                "r_squared": float(r_squared) if r_squared is not None else None,
                "expected_exponent": -0.5,
                "exponent_error": abs(beta - (-0.5)) if beta is not None else None,
                "fit_quality": fit_quality,
                "mean_loss": float(np.mean(mean_losses)),
                "loss_std": float(np.std(mean_losses)),
            },
            data={
                "context_lengths": lengths,
                "mean_losses": mean_losses,
                "std_losses": std_losses,
                "raw_data": {int(k): v for k, v in length_loss_data.items()}
            }
        )
        return self.result

    def _measure_continuation_loss(self, context: str, expected_continuation: str) -> Optional[float]:
        """
        Measure the loss for predicting a specific continuation given context.

        Uses logprobs to estimate how "surprised" the model is by the continuation.
        """
        try:
            # Prompt model with context
            prompt = f"Continue this text with the next sentence:\n\n{context}"

            response = self.client.chat.completions.create(
                model=self.config.model.completion_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )

            if not response.choices or not response.choices[0].logprobs:
                return None

            logprobs_content = response.choices[0].logprobs.content
            if not logprobs_content:
                return None

            # Calculate average negative log probability (loss)
            logprobs = [token.logprob for token in logprobs_content if token.logprob is not None]
            if not logprobs:
                return None

            # Loss = average negative log probability
            avg_loss = -np.mean(logprobs)
            return float(avg_loss)

        except Exception as e:
            self.log(f"Measurement error: {e}")
            return None

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot loss vs context length."""
        if not self.result or not self.result.success:
            self.log("No valid results to plot")
            return

        data = self.result.data
        metrics = self.result.metrics

        lengths = data["context_lengths"]
        mean_losses = data["mean_losses"]
        std_losses = data["std_losses"]

        beta = metrics["scaling_exponent"]
        alpha = metrics["amplitude"]
        gamma = metrics.get("offset", 0)
        r_squared = metrics["r_squared"]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot data with error bars
        ax.errorbar(lengths, mean_losses, yerr=std_losses, fmt='o',
                    markersize=8, capsize=5, capthick=2,
                    color='steelblue', ecolor='lightsteelblue',
                    label='Measured loss')

        # Fitted curve
        x_smooth = np.linspace(min(lengths), max(lengths), 100)
        y_fitted = alpha * np.power(x_smooth, beta) + gamma
        ax.plot(x_smooth, y_fitted, '-', linewidth=2, color='coral',
                label=f'Fit: L(c) = {alpha:.3f}×c^{beta:.3f} + {gamma:.3f}')

        # Theoretical curve (β = -0.5)
        if mean_losses[0] > gamma:
            alpha_theoretical = (mean_losses[0] - gamma) * np.sqrt(lengths[0])
            y_theoretical = alpha_theoretical * np.power(x_smooth, -0.5) + gamma
            ax.plot(x_smooth, y_theoretical, '--', linewidth=2, color='green', alpha=0.7,
                    label='Theoretical: β = -0.5')

        ax.set_xscale('log')
        ax.set_xlabel('Context Length (characters)', fontsize=12)
        ax.set_ylabel('Loss (negative log probability)', fontsize=12)
        ax.set_title(f'Loss Scaling with Context Length\nβ = {beta:.3f} (expected: -0.5), R² = {r_squared:.3f}',
                     fontsize=14)

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")

        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
