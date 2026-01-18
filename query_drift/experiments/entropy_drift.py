"""
Entropy Drift Experiment - Control for LayerNorm artifact.

This experiment measures the entropy of the output distribution directly
rather than just embedding vector geometry. This controls for the argument
that embedding variance saturation is merely a LayerNorm constraint.

If attention was drifting via Brownian motion, the entropy of the output
distribution would increase linearly. Instead, we predict entropy saturation
(OU process behavior).
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.math import fit_power_law


class EntropyDriftExperiment(BaseExperiment):
    """
    Experiment to measure entropy drift in model outputs over generation.
    
    Key insight: By measuring logit entropy rather than embedding variance,
    we bypass LayerNorm constraints and measure attention drift directly.
    
    Predictions:
    - Brownian motion: Entropy grows linearly with time
    - OU process (COFFEE Law): Entropy saturates to a plateau
    """
    
    experiment_name = "entropy_drift"
    
    def run(self) -> ExperimentResult:
        """Execute the entropy drift experiment."""
        self.log("Starting ENTROPY DRIFT experiment...")
        self.log("This controls for LayerNorm artifact by measuring output entropy directly")
        
        # Configuration
        num_sequences = 30  # Number of sequences to generate
        max_tokens = 150    # Maximum generation length
        temperature = 1.0   # Keep at 1.0 for proper entropy measurement
        sample_positions = [10, 20, 30, 50, 70, 100, 130, 150]
        
        # Prefix for generation
        prefix = (
            "The development of artificial intelligence has progressed through "
            "several important phases, starting with early theoretical work in the "
            "1950s and continuing through modern deep learning breakthroughs."
        )
        
        self.log(f"Generating {num_sequences} sequences up to {max_tokens} tokens")
        self.log(f"Measuring entropy at positions: {sample_positions}")
        
        # Generate sequences with logprobs
        entropies_by_position = {pos: [] for pos in sample_positions}
        mean_logprobs_by_position = {pos: [] for pos in sample_positions}
        
        for seq_idx in range(num_sequences):
            self.log(f"  Sequence {seq_idx + 1}/{num_sequences}...")
            
            # Generate with logprobs
            response = self.client.chat.completions.create(
                model=self.config.model.completion_model,
                messages=[
                    {"role": "system", "content": "Continue the text naturally."},
                    {"role": "user", "content": prefix}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=20,  # Get top 20 logprobs for entropy calculation
                n=1
            )
            
            # Extract logprobs
            logprobs_data = response.choices[0].logprobs
            
            if logprobs_data is None or logprobs_data.content is None:
                self.log(f"    Warning: No logprobs for sequence {seq_idx + 1}")
                continue
            
            # Calculate entropy at each position
            token_positions = []
            token_entropies = []
            token_mean_logprobs = []
            
            for token_idx, token_data in enumerate(logprobs_data.content):
                position = token_idx + 1  # 1-indexed
                
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    # Extract logprobs from top_logprobs
                    logprobs = []
                    for top_token in token_data.top_logprobs:
                        if hasattr(top_token, 'logprob'):
                            logprobs.append(top_token.logprob)
                    
                    if logprobs:
                        # Convert to probabilities
                        probs = np.exp(logprobs)
                        probs = probs / np.sum(probs)  # Normalize
                        
                        # Calculate entropy: H = -sum(p * log(p))
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        mean_logprob = np.mean(logprobs)
                        
                        token_positions.append(position)
                        token_entropies.append(entropy)
                        token_mean_logprobs.append(mean_logprob)
            
            # Store entropies at sample positions
            for pos in sample_positions:
                if pos <= len(token_entropies):
                    entropies_by_position[pos].append(token_entropies[pos - 1])
                    mean_logprobs_by_position[pos].append(token_mean_logprobs[pos - 1])
        
        # Aggregate statistics
        positions = []
        mean_entropies = []
        std_entropies = []
        mean_logprobs = []
        
        for pos in sample_positions:
            if len(entropies_by_position[pos]) >= 3:
                positions.append(pos)
                mean_ent = np.mean(entropies_by_position[pos])
                std_ent = np.std(entropies_by_position[pos])
                mean_lp = np.mean(mean_logprobs_by_position[pos])
                
                mean_entropies.append(mean_ent)
                std_entropies.append(std_ent)
                mean_logprobs.append(mean_lp)
                
                self.log(f"  Position {pos}: Entropy={mean_ent:.4f} ±{std_ent:.4f}")
        
        positions = np.array(positions)
        mean_entropies = np.array(mean_entropies)
        std_entropies = np.array(std_entropies)
        
        # Fit models
        # 1. Linear fit (Brownian prediction): entropy ~ position
        linear_fit = self._fit_linear(positions, mean_entropies)
        
        # 2. Saturation fit (OU prediction): entropy ~ a * (1 - exp(-k*position)) + offset
        saturation_fit = self._fit_saturation(positions, mean_entropies)
        
        # Determine which model fits better
        better_model = None
        if linear_fit and saturation_fit:
            if saturation_fit['r_squared'] > linear_fit['r_squared']:
                better_model = "saturation"
                self.log("✓ Saturation model fits better (supports OU/COFFEE Law)")
            else:
                better_model = "linear"
                self.log("✓ Linear model fits better (supports Brownian motion)")
        
        # Test for saturation: does entropy plateau?
        saturation_detected = False
        if len(mean_entropies) >= 3:
            # Compare entropy change in first half vs second half
            mid = len(mean_entropies) // 2
            first_half_change = mean_entropies[mid] - mean_entropies[0]
            second_half_change = mean_entropies[-1] - mean_entropies[mid]
            
            # If second half change is < 50% of first half, consider saturated
            if first_half_change > 0 and second_half_change < 0.5 * first_half_change:
                saturation_detected = True
                self.log(f"✓ Entropy saturation detected (2nd half change: {second_half_change:.4f} < 50% of 1st half: {first_half_change:.4f})")
        
        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "num_sequences": num_sequences,
                "saturation_detected": saturation_detected,
                "better_model": better_model,
                "linear_slope": linear_fit['slope'] if linear_fit else None,
                "linear_r_squared": linear_fit['r_squared'] if linear_fit else None,
                "saturation_plateau": saturation_fit['plateau'] if saturation_fit else None,
                "saturation_rate": saturation_fit['rate'] if saturation_fit else None,
                "saturation_r_squared": saturation_fit['r_squared'] if saturation_fit else None,
                "layernorm_artifact_controlled": True,  # This experiment controls for it
            },
            data={
                "positions": positions,
                "mean_entropies": mean_entropies,
                "std_entropies": std_entropies,
                "mean_logprobs": mean_logprobs,
                "linear_fit": linear_fit,
                "saturation_fit": saturation_fit,
                "entropies_by_position": {k: v for k, v in entropies_by_position.items() if v},
            }
        )
        return self.result
    
    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Fit linear model: y = mx + b (Brownian prediction)."""
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "model": "linear"
            }
        except Exception as e:
            self.log(f"Linear fit failed: {e}")
            return None
    
    def _fit_saturation(self, x: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Fit saturation model: y = A * (1 - exp(-k*x)) + offset (OU prediction)."""
        try:
            from scipy.optimize import curve_fit
            
            def saturation_model(x, A, k, offset):
                return A * (1 - np.exp(-k * x)) + offset
            
            # Initial guess
            p0 = [y[-1] - y[0], 0.01, y[0]]
            
            popt, pcov = curve_fit(saturation_model, x, y, p0=p0, maxfev=10000)
            A, k, offset = popt
            
            y_pred = saturation_model(x, A, k, offset)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            plateau = A + offset  # Asymptotic value
            
            return {
                "amplitude": float(A),
                "rate": float(k),
                "offset": float(offset),
                "plateau": float(plateau),
                "r_squared": float(r_squared),
                "model": "saturation"
            }
        except Exception as e:
            self.log(f"Saturation fit failed: {e}")
            return None
    
    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot entropy vs position with model fits."""
        if not self.result or not self.result.success:
            self.log("No results to plot")
            return
        
        data = self.result.data
        metrics = self.result.metrics
        
        positions = np.array(data["positions"])
        mean_entropies = np.array(data["mean_entropies"])
        std_entropies = np.array(data["std_entropies"])
        linear_fit = data.get("linear_fit")
        saturation_fit = data.get("saturation_fit")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot data with error bars
        ax.errorbar(positions, mean_entropies, yerr=std_entropies,
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color='darkblue', ecolor='gray', label='Measured Entropy (±1σ)', zorder=3)
        
        # Generate smooth x for model curves
        x_smooth = np.linspace(positions[0], positions[-1], 200)
        
        # Linear fit
        if linear_fit:
            y_linear = linear_fit['slope'] * x_smooth + linear_fit['intercept']
            ax.plot(x_smooth, y_linear, 'r--', linewidth=2.5, alpha=0.8,
                   label=f"Linear (Brownian): slope={linear_fit['slope']:.5f}, R²={linear_fit['r_squared']:.3f}")
        
        # Saturation fit
        if saturation_fit:
            y_sat = saturation_fit['amplitude'] * (1 - np.exp(-saturation_fit['rate'] * x_smooth)) + saturation_fit['offset']
            ax.plot(x_smooth, y_sat, 'g-', linewidth=2.5, alpha=0.8,
                   label=f"Saturation (OU): plateau={saturation_fit['plateau']:.3f}, R²={saturation_fit['r_squared']:.3f}")
        
        ax.set_xlabel('Token Position', fontsize=13)
        ax.set_ylabel('Output Distribution Entropy H(p)', fontsize=13)
        ax.set_title('Entropy Drift: Control for LayerNorm Artifact', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotation
        better = metrics.get('better_model', 'unknown')
        saturated = "✓ Saturated" if metrics['saturation_detected'] else "✗ Not Saturated"
        text = f"Better Model: {better}\n{saturated}\nLayerNorm: Controlled"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
