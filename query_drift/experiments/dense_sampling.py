"""
Dense Sampling Experiment with Prompt Diversity.

Implements "Dense sampling + prompt diversity (hierarchical)" for the COFFEE Law
validation paper. This experiment uses:
1. Dense token sampling: fine-grained at early positions, logarithmic at later positions
2. Prompt diversity: 50-100 diverse prompts across different domains
3. Cross-prompt confidence intervals: variance computed ACROSS different prompts
"""

from typing import Optional
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.embeddings import EmbeddingClient
from ..utils.math import fit_power_law, PowerLawFit
from ..config import ExperimentConfig


# Diverse seed prompts spanning multiple domains and topics
DIVERSE_PROMPTS = [
    # Science & Technology
    "The discovery of quantum entanglement has revolutionized our understanding of",
    "Machine learning algorithms are increasingly being applied to solve problems in",
    "The James Webb Space Telescope has revealed unprecedented details about",
    "Advances in CRISPR gene editing technology have opened new possibilities for",
    "The development of renewable energy sources is critical because",
    "Nanotechnology applications in medicine are transforming how we approach",
    "The history of computer science begins with early pioneers who envisioned",
    "Climate models predict significant changes in global weather patterns due to",
    "Artificial neural networks are inspired by biological systems and can",
    "The search for extraterrestrial life has focused on planets within",

    # History & Culture
    "The Renaissance period marked a significant shift in European thinking about",
    "Ancient civilizations developed sophisticated systems for managing",
    "The Industrial Revolution transformed society by introducing",
    "Medieval manuscripts reveal fascinating insights into daily life during",
    "The Silk Road facilitated cultural exchange between East and West through",
    "Archaeological discoveries in Egypt continue to reshape our understanding of",
    "The French Revolution fundamentally altered concepts of citizenship and",
    "Indigenous cultures have preserved traditional knowledge about",
    "The Cold War era shaped international relations through",
    "Ancient Greek philosophy established foundational ideas about",

    # Arts & Literature
    "Shakespeare's influence on the English language extends far beyond",
    "Modern art movements challenged traditional notions of beauty by",
    "The evolution of jazz music reflects broader social changes in",
    "Classical literature continues to resonate with readers because",
    "Film as an art form developed rapidly during the early twentieth century when",
    "Poetry across cultures shares common themes of love, loss, and",
    "Architecture in major cities reflects the values and aspirations of",
    "Dance traditions around the world embody cultural expressions of",
    "The novel as a literary form emerged to explore questions about",
    "Photography transformed how people document and remember",

    # Philosophy & Psychology
    "The nature of consciousness remains one of the most profound mysteries in",
    "Ethical frameworks help us navigate complex decisions involving",
    "Memory and identity are interconnected in ways that suggest",
    "The concept of free will has been debated by philosophers who argue",
    "Cognitive biases affect our decision-making processes by",
    "The relationship between language and thought has implications for",
    "Existentialist philosophers grappled with questions about meaning and",
    "Social psychology research demonstrates how groups influence",
    "The philosophy of mind addresses fundamental questions about",
    "Moral reasoning develops throughout the lifespan as individuals learn",

    # Economics & Society
    "Global trade patterns have shifted dramatically in recent decades due to",
    "Urban planning challenges include balancing growth with",
    "The economics of healthcare systems varies significantly across",
    "Social media platforms have transformed how people communicate and",
    "Education systems around the world are adapting to prepare students for",
    "The gig economy has created new opportunities and challenges for",
    "Public policy decisions about infrastructure investment affect",
    "Income inequality has been rising in many countries because",
    "The future of work is being reshaped by automation and",
    "Environmental sustainability requires coordinated efforts among",

    # Nature & Environment
    "Biodiversity loss threatens ecosystem stability because species interact",
    "Ocean ecosystems are particularly vulnerable to changes in",
    "Forests play a crucial role in carbon sequestration by",
    "Wildlife conservation efforts have successfully restored populations of",
    "The water cycle affects weather patterns and agricultural productivity through",
    "Mountain ecosystems harbor unique species that have adapted to",
    "Coral reefs support incredible biodiversity but face threats from",
    "Migratory patterns of birds reveal important information about",
    "Soil health determines agricultural productivity and depends on",
    "Wetlands provide essential ecosystem services including",

    # Health & Medicine
    "The human immune system responds to pathogens through",
    "Mental health awareness has increased as society recognizes",
    "Nutrition science continues to evolve our understanding of",
    "Sleep research has revealed its critical importance for",
    "Epidemiology helps track disease outbreaks by analyzing",
    "The microbiome influences health in ways that researchers are only beginning",
    "Preventive medicine focuses on reducing disease risk through",
    "Medical imaging technologies have advanced diagnosis by allowing",
    "The placebo effect demonstrates the powerful connection between",
    "Genetic factors interact with environment to influence",

    # Mathematics & Logic
    "Mathematical patterns appear throughout nature in examples like",
    "Statistical reasoning helps us make sense of uncertain data by",
    "The history of mathematics reveals how different cultures contributed",
    "Game theory provides insights into strategic decision-making when",
    "Cryptography relies on mathematical principles to secure",
    "Probability theory allows us to quantify uncertainty in",
    "Geometric principles underlie many aspects of design and",
    "The concept of infinity has fascinated mathematicians who explore",
    "Algorithms are step-by-step procedures that solve problems by",
    "Mathematical modeling helps predict complex system behavior through",

    # Space & Astronomy
    "Black holes represent extreme conditions where gravity becomes",
    "The formation of solar systems begins with clouds of gas that",
    "Exoplanet discoveries have revealed diverse worlds including",
    "The expansion of the universe suggests that space itself is",
    "Stellar evolution determines how stars change throughout their",
    "Dark matter and dark energy comprise most of the universe yet remain",
    "Space exploration has yielded technological innovations including",
    "The search for habitable exoplanets focuses on factors like",
    "Galaxies interact and merge over cosmic timescales through",
    "Cosmic radiation poses challenges for long-duration space missions because",

    # Food & Agriculture
    "Sustainable agriculture practices aim to maintain productivity while",
    "Food preservation techniques have evolved from ancient methods to",
    "The global food supply chain connects producers and consumers through",
    "Fermentation has been used across cultures to create foods like",
    "Crop diversity is essential for food security because",
    "Traditional cooking methods reflect local ingredients and",
    "Food waste reduction requires changes at every stage of",
    "Plant-based diets have gained popularity due to concerns about",
    "Agricultural technology innovations include precision farming that",
    "The cultural significance of shared meals extends beyond nutrition to",
]


class DenseSamplingExperiment(BaseExperiment):
    """
    Experiment implementing dense sampling with prompt diversity.

    This experiment validates the COFFEE Law predictions using:
    1. Hierarchical position sampling (dense early, logarithmic later)
    2. Diverse prompt trajectories (each prompt generates its own continuation)
    3. Cross-prompt confidence intervals for robust statistical inference

    Key insight: By computing variance ACROSS different prompts rather than
    within-prompt continuations, we get more independent samples and can
    compute meaningful confidence intervals for the Hurst exponent.
    """

    experiment_name = "dense_sampling"

    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[OpenAI] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        num_prompts: int = 50,
        max_tokens: int = 2000,
        dense_step: int = 15,
        dense_cutoff: int = 200,
        quick_mode: bool = False,
    ):
        """
        Initialize the dense sampling experiment.

        Args:
            config: Experiment configuration
            client: OpenAI client
            embedding_client: Embedding client
            num_prompts: Number of diverse prompts to use (50-100)
            max_tokens: Maximum tokens to generate per prompt
            dense_step: Step size for dense sampling in early positions (10-20)
            dense_cutoff: Position where dense sampling transitions to logarithmic
            quick_mode: If True, use reduced parameters for faster testing
        """
        super().__init__(config, client, embedding_client)

        if quick_mode:
            num_prompts = 10
            max_tokens = 500
            dense_step = 30
            dense_cutoff = 100

        self.num_prompts = min(num_prompts, len(DIVERSE_PROMPTS))
        self.max_tokens = max_tokens
        self.dense_step = dense_step
        self.dense_cutoff = dense_cutoff
        self._positions: np.ndarray = np.array([])
        self._variances: np.ndarray = np.array([])
        self._variance_cis: list[tuple[float, float]] = []
        self._fit_result: Optional[PowerLawFit] = None
        self._hurst_ci: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _generate_hierarchical_positions(self) -> np.ndarray:
        """
        Generate position indices with hierarchical/dense-then-log spacing.

        Returns:
            Array of token positions to sample:
            - Dense sampling (step 10-20) from 0 to dense_cutoff (~200 tokens)
            - Logarithmic spacing from dense_cutoff to max_tokens (~2000 tokens)
        """
        positions = []

        # Phase 1: Dense sampling for early positions (0 to dense_cutoff)
        # Use step size of dense_step (typically 10-20 tokens)
        dense_positions = list(range(
            self.dense_step,
            self.dense_cutoff + 1,
            self.dense_step
        ))
        positions.extend(dense_positions)

        # Phase 2: Logarithmic spacing for later positions
        # From dense_cutoff to max_tokens
        if self.max_tokens > self.dense_cutoff:
            # Generate approximately 20 log-spaced points
            num_log_points = 20
            log_positions = np.logspace(
                np.log10(self.dense_cutoff),
                np.log10(self.max_tokens),
                num_log_points
            )
            # Filter to only keep positions beyond dense sampling
            log_positions = [int(p) for p in log_positions if p > self.dense_cutoff]
            # Remove duplicates and sort
            log_positions = sorted(set(log_positions))
            positions.extend(log_positions)

        # Remove duplicates and sort
        positions = sorted(set(positions))

        self.log(f"Generated {len(positions)} hierarchical positions")
        self.log(f"  Dense phase: {len(dense_positions)} positions (step={self.dense_step})")
        self.log(f"  Log phase: {len(positions) - len(dense_positions)} positions")

        return np.array(positions)

    def _select_diverse_prompts(self) -> list[str]:
        """
        Select a diverse subset of prompts for the experiment.

        Returns:
            List of diverse seed prompts
        """
        # Randomly sample from the full set to ensure variety
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(
            len(DIVERSE_PROMPTS),
            size=self.num_prompts,
            replace=False
        )
        prompts = [DIVERSE_PROMPTS[i] for i in indices]

        self.log(f"Selected {len(prompts)} diverse prompts across domains")
        return prompts

    def _generate_continuation(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0
    ) -> str:
        """
        Generate a continuation from a seed prompt.

        Args:
            prompt: Seed prompt to continue
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated continuation text
        """
        response = self.client.chat.completions.create(
            model=self.config.model.completion_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Continue the given text naturally and in detail. "
                        "Write a thorough exploration of the topic. "
                        "Do not ask questions or add meta-commentary."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1
        )
        return response.choices[0].message.content or ""

    def _compute_bootstrap_ci(
        self,
        values: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval for the mean.

        Args:
            values: Array of measurements
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            (lower_bound, upper_bound) for the mean
        """
        n = len(values)
        if n < 2:
            return (float(values[0]), float(values[0])) if n == 1 else (0.0, 0.0)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def _bootstrap_hurst_ci(
        self,
        positions: np.ndarray,
        prompt_variances: dict[int, np.ndarray],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> tuple[float, float, float]:
        """
        Compute bootstrap CI for Hurst exponent across prompts.

        Args:
            positions: Array of token positions
            prompt_variances: Dict mapping position -> array of per-prompt variances
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            (lower, point_estimate, upper) for Hurst exponent
        """
        num_prompts = len(next(iter(prompt_variances.values())))
        hurst_estimates = []

        for _ in range(n_bootstrap):
            # Resample prompts
            prompt_indices = np.random.choice(num_prompts, size=num_prompts, replace=True)

            # Compute mean variance at each position from resampled prompts
            boot_variances = []
            for pos in positions:
                if pos in prompt_variances:
                    var_samples = prompt_variances[pos][prompt_indices]
                    boot_variances.append(np.mean(var_samples))

            boot_variances = np.array(boot_variances)

            # Fit power law
            try:
                fit = fit_power_law(positions[:len(boot_variances)], boot_variances)
                hurst = fit.exponent / 2.0
                if 0 < hurst < 2:  # Sanity check
                    hurst_estimates.append(hurst)
            except Exception:
                continue

        if len(hurst_estimates) < 10:
            # Fallback to point estimate
            mean_variances = np.array([np.mean(prompt_variances[pos]) for pos in positions])
            fit = fit_power_law(positions, mean_variances)
            point = fit.exponent / 2.0
            return (point, point, point)

        alpha = 1 - confidence
        lower = np.percentile(hurst_estimates, 100 * alpha / 2)
        upper = np.percentile(hurst_estimates, 100 * (1 - alpha / 2))
        point = np.median(hurst_estimates)

        return (float(lower), float(point), float(upper))

    def run(self) -> ExperimentResult:
        """Execute the dense sampling experiment with prompt diversity."""
        cfg = self.config.embedding_drift

        try:
            # Step 1: Generate hierarchical positions
            positions = self._generate_hierarchical_positions()

            # Step 2: Select diverse prompts
            prompts = self._select_diverse_prompts()

            # Step 3: Generate continuations for each prompt
            self.log(f"Generating continuations for {len(prompts)} prompts...")
            continuations = []
            for i, prompt in enumerate(prompts):
                continuation = self._generate_continuation(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=cfg.temperature
                )
                continuations.append((prompt, continuation))

                if self.config.output.verbose and (i + 1) % 10 == 0:
                    self.log(f"Generated {i + 1}/{len(prompts)} continuations")

            self.log(f"Generated {len(continuations)} prompt trajectories")

            # Step 4: Compute embeddings at each position for each prompt trajectory
            # Key: We compute variance ACROSS prompts, not within-prompt
            self.log("Computing embeddings at hierarchical positions...")

            # Store embeddings per position per prompt
            embeddings_by_position: dict[int, list[np.ndarray]] = {
                pos: [] for pos in positions
            }

            for prompt_idx, (prompt, continuation) in enumerate(continuations):
                # Tokenize by words (rough approximation)
                words = continuation.split()

                for pos in positions:
                    if pos <= len(words):
                        # Combine prompt + continuation[:pos]
                        text = prompt + " " + " ".join(words[:pos])
                    else:
                        # Use full continuation if shorter
                        text = prompt + " " + continuation

                    emb = self.embedding_client.get_embedding(text)
                    embeddings_by_position[pos].append(np.array(emb.embedding))

                if self.config.output.verbose and (prompt_idx + 1) % 10 == 0:
                    self.log(f"Processed embeddings for {prompt_idx + 1}/{len(continuations)} prompts")

            # Step 5: Compute variance ACROSS prompts at each position
            # Store per-prompt squared distances for bootstrap
            prompt_squared_distances: dict[int, np.ndarray] = {}
            valid_positions = []
            variances = []
            variance_cis = []

            for pos in positions:
                embs = embeddings_by_position[pos]
                if len(embs) < 2:
                    continue

                emb_matrix = np.array(embs)
                centroid = np.mean(emb_matrix, axis=0)

                # Squared L2 distance from centroid for each prompt
                sq_distances = np.sum((emb_matrix - centroid) ** 2, axis=1)

                # Store for bootstrap
                prompt_squared_distances[pos] = sq_distances

                # Mean variance across prompts
                var = np.mean(sq_distances)

                if var > 1e-10:
                    valid_positions.append(pos)
                    variances.append(var)

                    # Bootstrap CI for variance at this position
                    ci = self._compute_bootstrap_ci(sq_distances)
                    variance_cis.append(ci)

                    if self.config.output.verbose:
                        self.log(f"  Position {pos}: variance = {var:.6f} "
                                f"[{ci[0]:.6f}, {ci[1]:.6f}]")

            self._positions = np.array(valid_positions)
            self._variances = np.array(variances)
            self._variance_cis = variance_cis

            if len(self._positions) < 3:
                raise ValueError(f"Not enough valid positions: {len(self._positions)} < 3")

            # Step 6: Fit power law and compute Hurst exponent
            self._fit_result = fit_power_law(self._positions, self._variances)
            hurst_point = self._fit_result.exponent / 2.0

            # Step 7: Bootstrap CI for Hurst exponent across prompts
            self._hurst_ci = self._bootstrap_hurst_ci(
                self._positions,
                prompt_squared_distances,
                n_bootstrap=1000,
                confidence=0.95
            )

            self.log(f"\nPower law fit: sigma^2(t) = {self._fit_result.amplitude:.6f} * t^{self._fit_result.exponent:.4f}")
            self.log(f"R^2 = {self._fit_result.r_squared:.4f}")
            self.log(f"Hurst exponent H = {hurst_point:.4f}")
            self.log(f"95% CI for H: [{self._hurst_ci[0]:.4f}, {self._hurst_ci[2]:.4f}]")

            # Step 8: Validate prediction (H ~ 0.5 for standard Brownian motion)
            h_tolerance = 0.2
            prediction_validated = abs(hurst_point - 0.5) < h_tolerance
            ci_includes_half = self._hurst_ci[0] <= 0.5 <= self._hurst_ci[2]

            self.log(f"\nPrediction (H ~ 0.5):")
            self.log(f"  Point estimate check: {'PASS' if prediction_validated else 'FAIL'}")
            self.log(f"  CI includes 0.5: {'PASS' if ci_includes_half else 'FAIL'}")

            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=True,
                metrics={
                    "hurst_exponent": float(hurst_point),
                    "hurst_ci_lower": float(self._hurst_ci[0]),
                    "hurst_ci_upper": float(self._hurst_ci[2]),
                    "power_law_exponent": float(self._fit_result.exponent),
                    "r_squared": float(self._fit_result.r_squared),
                    "amplitude": float(self._fit_result.amplitude),
                    "prediction_validated": prediction_validated,
                    "ci_includes_half": ci_includes_half,
                    "num_prompts": len(prompts),
                    "num_positions": len(self._positions),
                },
                data={
                    "positions": self._positions.tolist(),
                    "variances": self._variances.tolist(),
                    "variance_cis": variance_cis,
                    "hurst_ci": self._hurst_ci,
                    "num_prompts": len(prompts),
                    "dense_step": self.dense_step,
                    "dense_cutoff": self.dense_cutoff,
                    "max_tokens": self.max_tokens,
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

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot variance trajectory with error bands and power law fit.

        Creates a log-log plot showing:
        - Variance measurements at each position with bootstrap CIs
        - Fitted power law curve
        - Theoretical H=0.5 reference line
        - Hurst exponent with confidence interval
        """
        if self.result is None or not self.result.success:
            self.log("Cannot plot: no successful experiment result")
            return

        positions = np.array(self.result.data["positions"])
        variances = np.array(self.result.data["variances"])
        variance_cis = self.result.data["variance_cis"]
        hurst = self.result.metrics["hurst_exponent"]
        hurst_ci_lower = self.result.metrics["hurst_ci_lower"]
        hurst_ci_upper = self.result.metrics["hurst_ci_upper"]
        r_squared = self.result.metrics["r_squared"]
        amplitude = self.result.metrics["amplitude"]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Error bars for variance CIs
        ci_lower = np.array([ci[0] for ci in variance_cis])
        ci_upper = np.array([ci[1] for ci in variance_cis])
        yerr = np.array([variances - ci_lower, ci_upper - variances])

        # Plot data points with error bars
        ax.errorbar(
            positions, variances, yerr=yerr,
            fmt='o', markersize=8, color='blue', alpha=0.7,
            ecolor='lightblue', elinewidth=2, capsize=3,
            label='Measured variance (95% CI across prompts)'
        )

        # Fitted power law
        pos_smooth = np.logspace(
            np.log10(positions.min()),
            np.log10(positions.max()),
            100
        )
        fitted_variance = amplitude * np.power(pos_smooth, 2 * hurst)
        ax.plot(
            pos_smooth, fitted_variance, 'b-', linewidth=2, alpha=0.8,
            label=f'Fitted: $\\sigma^2 \\propto t^{{{2*hurst:.2f}}}$ (H={hurst:.3f})'
        )

        # Confidence band for power law fit (using Hurst CI)
        fitted_lower = amplitude * np.power(pos_smooth, 2 * hurst_ci_lower)
        fitted_upper = amplitude * np.power(pos_smooth, 2 * hurst_ci_upper)
        ax.fill_between(
            pos_smooth, fitted_lower, fitted_upper,
            color='blue', alpha=0.1,
            label=f'95% CI for H: [{hurst_ci_lower:.3f}, {hurst_ci_upper:.3f}]'
        )

        # Theoretical H=0.5 reference
        if variances[0] > 0 and positions[0] > 0:
            theoretical_amp = variances[0] / positions[0]
            theoretical_var = theoretical_amp * pos_smooth
            ax.plot(
                pos_smooth, theoretical_var, 'r--', linewidth=2, alpha=0.7,
                label='Theoretical: $\\sigma^2 \\propto t$ (H=0.5)'
            )

        # Mark transition from dense to log sampling
        ax.axvline(
            x=self.dense_cutoff, color='gray', linestyle=':', alpha=0.5,
            label=f'Dense/Log transition ({self.dense_cutoff} tokens)'
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Continuation Position (tokens)', fontsize=12)
        ax.set_ylabel('Embedding Variance $\\sigma^2(t)$', fontsize=12)
        ax.set_title(
            'Dense Sampling: Variance Growth with Prompt Diversity\n'
            f'({self.num_prompts} diverse prompts, hierarchical sampling)',
            fontsize=14
        )

        # Statistics annotation
        textstr = '\n'.join([
            f'Hurst H = {hurst:.3f}',
            f'95% CI: [{hurst_ci_lower:.3f}, {hurst_ci_upper:.3f}]',
            f'$R^2$ = {r_squared:.4f}',
            f'N prompts = {self.num_prompts}'
        ])
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")
        elif self.config.output.show_plots:
            plt.show()

        plt.close(fig)
