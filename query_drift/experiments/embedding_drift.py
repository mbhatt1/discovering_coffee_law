"""
Embedding Drift Experiment.

Validates that embedding variance grows as sigma^2(t) ~ t^(2H),
measuring drift in the continuation portion (not the fixed prefix).
"""

from typing import Optional
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.embeddings import EmbeddingClient
from ..utils.math import fit_power_law, PowerLawFit
from ..config import ExperimentConfig


class EmbeddingDriftExperiment(BaseExperiment):
    """
    Experiment to measure embedding drift and estimate Hurst exponent.

    Key insight: We measure variance in the CONTINUATION embeddings,
    not from the start of text. This ensures we're measuring drift
    from the divergence point.
    """

    experiment_name = "embedding_drift"

    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[OpenAI] = None,
        embedding_client: Optional[EmbeddingClient] = None
    ):
        super().__init__(config, client, embedding_client)
        self._continuations: list[str] = []
        self._prefix: str = ""
        self._fit_result: Optional[PowerLawFit] = None

    def run(self) -> ExperimentResult:
        """Execute the embedding drift experiment."""
        cfg = self.config.embedding_drift

        try:
            # Step 1: Define baseline prefix (matching bulletproof entropy_drift experiment)
            self._prefix = (
                "The development of artificial intelligence has progressed through "
                "several important phases, starting with early theoretical work in the "
                "1950s and continuing through modern deep learning breakthroughs."
            )
            self.log(f"Using prefix: '{self._prefix[:50]}...'")

            # Step 2: Generate multiple diverse continuations
            self.log(f"Generating {cfg.num_continuations} continuations...")
            raw_continuations = self._generate_continuations(
                prefix=self._prefix,
                num_continuations=cfg.num_continuations,
                max_tokens=cfg.max_tokens_drift,
                temperature=cfg.temperature
            )
            self.log(f"Generated {len(raw_continuations)} continuations")

            # Step 3: Compute embeddings at different positions IN THE CONTINUATION
            # Positions are relative to start of continuation, not prefix
            positions_to_sample = [p for p in cfg.sample_positions if p <= cfg.max_tokens_drift]
            if len(positions_to_sample) < 3:
                positions_to_sample = [10, 20, 40, 60, 80, 100]
                positions_to_sample = [p for p in positions_to_sample if p <= cfg.max_tokens_drift]

            self.log(f"Computing embeddings at continuation positions: {positions_to_sample}")

            # Compute embeddings for prefix + continuation[0:pos] for each continuation
            embeddings_by_position = {pos: [] for pos in positions_to_sample}

            for i, continuation in enumerate(raw_continuations):
                # Split continuation into words (rough tokenization)
                words = continuation.split()

                for pos in positions_to_sample:
                    if pos <= len(words):
                        # Take first 'pos' words of the continuation
                        text = self._prefix + " " + " ".join(words[:pos])
                    else:
                        # Use full continuation if shorter than position
                        text = self._prefix + " " + continuation

                    emb = self.embedding_client.get_embedding(text)
                    embeddings_by_position[pos].append(np.array(emb.embedding))

                if self.config.output.verbose and (i + 1) % 5 == 0:
                    self.log(f"Processed {i + 1}/{len(raw_continuations)} continuations")

            # Step 4: Compute variance at each position
            # Variance = mean squared distance from centroid
            positions = []
            variances = []

            for pos in positions_to_sample:
                embs = embeddings_by_position[pos]
                if len(embs) < 2:
                    continue

                emb_matrix = np.array(embs)
                centroid = np.mean(emb_matrix, axis=0)

                # Mean squared L2 distance from centroid
                distances_sq = np.sum((emb_matrix - centroid) ** 2, axis=1)
                var = np.mean(distances_sq)

                if var > 1e-10:  # Skip if essentially zero
                    positions.append(pos)
                    variances.append(var)
                    self.log(f"  Position {pos}: variance = {var:.6f}")

            positions = np.array(positions)
            variances = np.array(variances)

            if len(positions) < 3:
                raise ValueError(f"Not enough valid data points: {len(positions)} < 3")

            # Step 5: Fit power law: sigma^2(t) = a * t^(2H)
            self._fit_result = fit_power_law(positions, variances)

            # Hurst exponent H = exponent / 2
            hurst_exponent = self._fit_result.exponent / 2.0

            self.log(f"Power law fit: σ²(t) = {self._fit_result.amplitude:.6f} × t^{self._fit_result.exponent:.4f}")
            self.log(f"R² = {self._fit_result.r_squared:.4f}")
            self.log(f"Estimated Hurst exponent H = {hurst_exponent:.4f}")

            # Validate prediction
            h_tolerance = 0.2
            prediction_validated = abs(hurst_exponent - 0.5) < h_tolerance
            self.log(f"Prediction (H~0.5): {'PASS' if prediction_validated else 'FAIL'}")

            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=True,
                metrics={
                    "hurst_exponent": float(hurst_exponent),
                    "power_law_exponent": float(self._fit_result.exponent),
                    "r_squared": float(self._fit_result.r_squared),
                    "amplitude": float(self._fit_result.amplitude),
                    "prediction_validated": prediction_validated,
                },
                data={
                    "positions": positions,
                    "variances": variances,
                    "num_continuations": len(raw_continuations),
                    "prefix": self._prefix,
                }
            )
            return self.result

        except Exception as e:
            self.log(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=False,
                error=str(e)
            )
            return self.result

    def _generate_continuations(
        self,
        prefix: str,
        num_continuations: int,
        max_tokens: int,
        temperature: float
    ) -> list[str]:
        """Generate diverse continuations from a prefix."""
        continuations = []

        for i in range(num_continuations):
            response = self.client.chat.completions.create(
                model=self.config.model.completion_model,
                messages=[
                    {"role": "system", "content": "Continue the given text naturally and creatively. Write diverse content."},
                    {"role": "user", "content": prefix}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            continuation = response.choices[0].message.content or ""
            continuations.append(continuation)

            if self.config.output.verbose and (i + 1) % 5 == 0:
                self.log(f"Generated {i + 1}/{num_continuations} continuations")

        return continuations

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot variance vs position on log-log scale."""
        if self.result is None or not self.result.success:
            self.log("Cannot plot: no successful experiment result")
            return

        positions = np.array(self.result.data["positions"])
        variances = np.array(self.result.data["variances"])
        hurst = self.result.metrics["hurst_exponent"]
        r_squared = self.result.metrics["r_squared"]
        amplitude = self.result.metrics["amplitude"]

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot data points
        ax.scatter(positions, variances, s=100, c="blue", alpha=0.7, zorder=3,
                   label="Measured variance")

        # Fitted power law
        pos_smooth = np.linspace(positions.min(), positions.max(), 100)
        fitted_variance = amplitude * np.power(pos_smooth, 2 * hurst)
        ax.plot(pos_smooth, fitted_variance, "b-", linewidth=2, alpha=0.8,
                label=f"Fitted: σ² ∝ t^{2*hurst:.2f} (H={hurst:.3f})")

        # Theoretical H=0.5 reference
        if variances[0] > 0 and positions[0] > 0:
            theoretical_amp = variances[0] / positions[0]
            theoretical_var = theoretical_amp * pos_smooth
            ax.plot(pos_smooth, theoretical_var, "r--", linewidth=2, alpha=0.7,
                    label="Theoretical: σ² ∝ t (H=0.5)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Continuation Position (tokens)", fontsize=12)
        ax.set_ylabel("Embedding Variance σ²(t)", fontsize=12)
        ax.set_title("Embedding Drift: Variance Growth", fontsize=14)

        # Annotation
        textstr = f"H = {hurst:.3f}\nR² = {r_squared:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.log(f"Plot saved to {save_path}")
        elif self.config.output.show_plots:
            plt.show()

        plt.close(fig)
