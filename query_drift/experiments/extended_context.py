"""
Extended Context Window Experiment - Test OU saturation at 10k+ tokens.

This experiment pushes context length to 10,000+ tokens to validate that
variance saturation (OU process) holds beyond the "short context regime"
and is not just an artifact of staying within the model's prime training window.

Note: Due to embedding model token limits (8191 for text-embedding-3-small),
we use a sliding window approach for contexts >8k tokens, measuring variance
in the model's representation of the most recent 7500 words. This tests whether
the model's output stability (measured via embedding variance) saturates even
when processing extremely long contexts.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import time

from .base import BaseExperiment, ExperimentResult
from ..utils.math import fit_power_law


class ExtendedContextExperiment(BaseExperiment):
    """
    Test embedding variance and alignment at extreme context lengths (up to 10k-16k tokens).
    
    This addresses the critique that 2.4k tokens is "short context" and the observed
    saturation might just be stable behavior within the model's prime training distribution.
    
    Predictions:
    - OU process: Variance saturates even at 10k+ tokens
    - Brownian motion: Variance continues to grow linearly
    """
    
    experiment_name = "extended_context"
    
    def run(self) -> ExperimentResult:
        """Execute the extended context experiment."""
        self.log("Starting EXTENDED CONTEXT experiment...")
        self.log("Testing variance saturation at 10k+ token contexts")
        
        # Extended context lengths: push beyond typical training distribution
        context_lengths = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000]
        num_trials = 3  # Balance between thoroughness and API cost
        num_samples_per_length = 10  # Number of continuations per context length
        
        self.log(f"Context lengths to test: {context_lengths}")
        self.log(f"Trials: {num_trials}, Samples per length: {num_samples_per_length}")
        
        # Storage for results
        all_trials = []
        
        for trial_idx in range(num_trials):
            self.log(f"\n=== Trial {trial_idx + 1}/{num_trials} ===")
            trial_result = self._run_single_trial(
                context_lengths=context_lengths,
                num_samples=num_samples_per_length,
                trial_idx=trial_idx
            )
            all_trials.append(trial_result)
        
        # Aggregate results
        return self._compile_results(all_trials, context_lengths, num_trials)
    
    def _run_single_trial(
        self,
        context_lengths: List[int],
        num_samples: int,
        trial_idx: int
    ) -> dict:
        """Run a single trial across all context lengths."""
        trial_results = {
            'context_lengths': [],
            'variances': [],
            'mean_cosine_similarities': [],
        }
        
        for ctx_len in context_lengths:
            self.log(f"  Testing context length: {ctx_len} tokens...")
            
            # Generate base context of appropriate length
            base_context = self._generate_long_context(target_tokens=ctx_len)
            actual_token_count = len(base_context.split())  # Rough token count
            
            self.log(f"    Generated context with ~{actual_token_count} words")
            
            # Generate multiple continuations from this context
            continuation_embeddings = []
            base_embedding = None
            
            for sample_idx in range(num_samples):
                # Generate a continuation
                response = self.client.chat.completions.create(
                    model=self.config.model.completion_model,
                    messages=[
                        {"role": "system", "content": "Continue the text naturally."},
                        {"role": "user", "content": base_context + " Therefore,"}
                    ],
                    max_tokens=50,
                    temperature=1.0,
                    n=1
                )
                
                continuation = response.choices[0].message.content or ""
                full_text = base_context + " Therefore, " + continuation
                
                # Truncate to embedding model's token limit (8191 for text-embedding-3-small)
                # We keep the last ~7500 tokens to stay safely within limit
                words = full_text.split()
                if len(words) > 7500:
                    # Keep the last 7500 words (roughly 10k tokens with overhead)
                    full_text = " ".join(words[-7500:])
                
                # Get embedding
                emb = self.embedding_client.get_embedding(full_text)
                emb_vec = np.array(emb.embedding)
                continuation_embeddings.append(emb_vec)
                
                # Store first embedding as base for similarity
                if sample_idx == 0:
                    base_embedding = emb_vec
                
                if sample_idx % 3 == 0:
                    self.log(f"      Sample {sample_idx + 1}/{num_samples}")
            
            # Calculate variance across continuations
            emb_matrix = np.array(continuation_embeddings)
            centroid = np.mean(emb_matrix, axis=0)
            distances_sq = np.sum((emb_matrix - centroid) ** 2, axis=1)
            variance = float(np.mean(distances_sq))
            
            # Calculate mean cosine similarity to base
            similarities = []
            for emb in continuation_embeddings[1:]:  # Skip first (itself)
                sim = np.dot(base_embedding, emb) / (
                    np.linalg.norm(base_embedding) * np.linalg.norm(emb)
                )
                similarities.append(sim)
            mean_similarity = float(np.mean(similarities)) if similarities else 1.0
            
            trial_results['context_lengths'].append(ctx_len)
            trial_results['variances'].append(variance)
            trial_results['mean_cosine_similarities'].append(mean_similarity)
            
            self.log(f"    Variance: {variance:.6f}, Mean Similarity: {mean_similarity:.4f}")
        
        return trial_results
    
    def _generate_long_context(self, target_tokens: int) -> str:
        """Generate a long context of approximately target_tokens length."""
        # Start with a seed
        context = (
            "The following is a comprehensive historical review of technological "
            "developments and their societal impacts over the past century. "
        )
        
        current_tokens = len(context.split())
        
        # Keep generating until we reach target
        while current_tokens < target_tokens:
            remaining = target_tokens - current_tokens
            chunk_size = min(500, remaining)  # Generate in chunks
            
            response = self.client.chat.completions.create(
                model=self.config.model.completion_model,
                messages=[
                    {"role": "system", "content": "Continue writing a detailed historical review. Be comprehensive and detailed."},
                    {"role": "user", "content": context}
                ],
                max_tokens=chunk_size,
                temperature=0.7,
                n=1
            )
            
            continuation = response.choices[0].message.content or ""
            context += " " + continuation
            current_tokens = len(context.split())
            
            if current_tokens % 1000 < chunk_size:
                self.log(f"      Generated ~{current_tokens} tokens so far...")
        
        return context
    
    def _compile_results(
        self,
        all_trials: List[dict],
        context_lengths: List[int],
        num_trials: int
    ) -> ExperimentResult:
        """Aggregate results across trials."""
        self.log(f"\n=== Aggregating {num_trials} trials ===")
        
        # Aggregate by context length
        aggregated = {
            ctx_len: {'variances': [], 'similarities': []}
            for ctx_len in context_lengths
        }
        
        for trial in all_trials:
            for i, ctx_len in enumerate(trial['context_lengths']):
                aggregated[ctx_len]['variances'].append(trial['variances'][i])
                aggregated[ctx_len]['similarities'].append(trial['mean_cosine_similarities'][i])
        
        # Compute statistics
        mean_variances = []
        std_variances = []
        mean_similarities = []
        
        for ctx_len in context_lengths:
            mean_var = np.mean(aggregated[ctx_len]['variances'])
            std_var = np.std(aggregated[ctx_len]['variances'])
            mean_sim = np.mean(aggregated[ctx_len]['similarities'])
            
            mean_variances.append(mean_var)
            std_variances.append(std_var)
            mean_similarities.append(mean_sim)
            
            self.log(f"  {ctx_len:5d} tokens: Variance={mean_var:.6f} ±{std_var:.6f}")
        
        # Fit models
        ctx_array = np.array(context_lengths)
        var_array = np.array(mean_variances)
        
        # Linear fit (Brownian)
        linear_fit = self._fit_linear(ctx_array, var_array)
        
        # Saturation fit (OU)
        saturation_fit = self._fit_saturation(ctx_array, var_array)
        
        # Determine which fits better
        better_model = None
        if linear_fit and saturation_fit:
            if saturation_fit['r_squared'] > linear_fit['r_squared']:
                better_model = "saturation"
                self.log("✓ Saturation model fits better (supports OU/COFFEE Law)")
            else:
                better_model = "linear"
                self.log("✓ Linear model fits better (supports Brownian motion)")
        
        # Test for saturation behavior
        saturation_detected = False
        if len(var_array) >= 4:
            # Compare growth rate in first quarter vs last quarter
            quarter = len(var_array) // 4
            first_growth = var_array[quarter] - var_array[0]
            last_growth = var_array[-1] - var_array[-1-quarter]
            
            if first_growth > 0 and last_growth < 0.5 * first_growth:
                saturation_detected = True
                self.log(f"✓ Saturation detected at long context (last quarter growth: {last_growth:.6f} < 50% of first: {first_growth:.6f})")
        
        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "num_trials": num_trials,
                "max_context_length": max(context_lengths),
                "saturation_detected": saturation_detected,
                "better_model": better_model,
                "linear_r_squared": linear_fit['r_squared'] if linear_fit else None,
                "saturation_r_squared": saturation_fit['r_squared'] if saturation_fit else None,
                "saturation_plateau": saturation_fit['plateau'] if saturation_fit else None,
                "extends_beyond_training": max(context_lengths) > 4000,  # Beyond typical training window
            },
            data={
                "context_lengths": context_lengths,
                "mean_variances": mean_variances,
                "std_variances": std_variances,
                "mean_similarities": mean_similarities,
                "linear_fit": linear_fit,
                "saturation_fit": saturation_fit,
                "all_trials": all_trials,
            }
        )
        return self.result
    
    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Fit linear model."""
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "model": "linear"
            }
        except Exception as e:
            self.log(f"Linear fit failed: {e}")
            return None
    
    def _fit_saturation(self, x: np.ndarray, y: np.ndarray) -> Optional[dict]:
        """Fit saturation model."""
        try:
            from scipy.optimize import curve_fit
            
            def saturation_model(x, A, k, offset):
                return A * (1 - np.exp(-k * x)) + offset
            
            p0 = [y[-1] - y[0], 0.0001, y[0]]
            popt, pcov = curve_fit(saturation_model, x, y, p0=p0, maxfev=10000)
            A, k, offset = popt
            
            y_pred = saturation_model(x, A, k, offset)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            plateau = A + offset
            
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
        """Plot variance vs context length."""
        if not self.result or not self.result.success:
            self.log("No results to plot")
            return
        
        data = self.result.data
        metrics = self.result.metrics
        
        context_lengths = np.array(data["context_lengths"])
        mean_variances = np.array(data["mean_variances"])
        std_variances = np.array(data["std_variances"])
        linear_fit = data.get("linear_fit")
        saturation_fit = data.get("saturation_fit")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot data
        ax.errorbar(context_lengths, mean_variances, yerr=std_variances,
                   fmt='o', markersize=10, capsize=5, capthick=2,
                   color='darkviolet', ecolor='gray', label='Measured Variance (±1σ)', zorder=3)
        
        # Smooth x for model curves
        x_smooth = np.linspace(context_lengths[0], context_lengths[-1], 200)
        
        # Linear fit
        if linear_fit:
            y_linear = linear_fit['slope'] * x_smooth + linear_fit['intercept']
            ax.plot(x_smooth, y_linear, 'r--', linewidth=2.5, alpha=0.8,
                   label=f"Linear (Brownian): R²={linear_fit['r_squared']:.3f}")
        
        # Saturation fit
        if saturation_fit:
            y_sat = saturation_fit['amplitude'] * (1 - np.exp(-saturation_fit['rate'] * x_smooth)) + saturation_fit['offset']
            ax.plot(x_smooth, y_sat, 'g-', linewidth=2.5, alpha=0.8,
                   label=f"Saturation (OU): plateau={saturation_fit['plateau']:.6f}, R²={saturation_fit['r_squared']:.3f}")
            
            # Draw plateau line
            ax.axhline(y=saturation_fit['plateau'], color='green', linestyle=':', 
                      alpha=0.5, linewidth=2, label='Plateau Level')
        
        ax.set_xlabel('Context Length (tokens)', fontsize=13)
        ax.set_ylabel('Embedding Variance σ²', fontsize=13)
        ax.set_title('Extended Context: Variance Saturation at 10k+ Tokens', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotation
        better = metrics.get('better_model', 'unknown')
        saturated = "✓ Saturated" if metrics['saturation_detected'] else "✗ Not Saturated"
        max_ctx = metrics['max_context_length']
        text = f"Max Context: {max_ctx:,} tokens\nBetter Model: {better}\n{saturated}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
