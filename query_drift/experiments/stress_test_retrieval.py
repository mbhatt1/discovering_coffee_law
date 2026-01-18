"""
Stress Test Retrieval Experiment - Extended "Lost in the Middle" validation.

This experiment pushes retrieval to extreme scales (200, 500, 1000+ distractors)
to break the 100% ceiling and measure degradation curves following OU predictions.
"""

import time
import numpy as np
from typing import Optional, List, Dict
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.math import fit_power_law

# Extended fact set for larger scale testing
EXTENDED_FACTS = [
    ("My favorite color is blue.", "What is my favorite color?", "blue"),
    ("I was born in Seattle, Washington.", "Where was I born?", "seattle"),
    ("My dog's name is Max.", "What is my pet's name?", "max"),
    ("I work as a software engineer at Google.", "What is my job?", "software engineer"),
    ("My favorite food is pizza with pepperoni.", "What food do I like?", "pizza"),
    ("I drive a red Honda Civic.", "What car do I drive?", "honda"),
    ("My best friend Sarah lives nearby.", "Who is my best friend?", "sarah"),
    ("I graduated from Stanford University in 2015.", "Where did I study?", "stanford"),
    ("My phone number ends with 4567.", "What's the end of my phone number?", "4567"),
    ("I love playing chess on weekends.", "What game do I play?", "chess"),
    ("I live in apartment 302.", "What's my apartment number?", "302"),
    ("I prefer drinking tea over coffee.", "Tea or coffee preference?", "tea"),
    ("My sister Emma lives in Boston.", "Where does my sister live?", "boston"),
    ("I started learning guitar last year.", "What instrument do I play?", "guitar"),
    ("I go to Planet Fitness gym.", "What gym do I go to?", "planet fitness"),
    ("I'm afraid of heights.", "What am I scared of?", "heights"),
    ("My favorite movie is Inception.", "What's my favorite movie?", "inception"),
    ("I'm allergic to shellfish and peanuts.", "What allergies do I have?", "shellfish"),
    ("I see the dentist every Tuesday.", "When's my dentist appointment?", "tuesday"),
    ("I collect vintage postcards from Europe.", "What do I collect?", "postcards"),
    ("My cat's name is Whiskers.", "What is my cat named?", "whiskers"),
    ("I was born on March 15th.", "When is my birthday?", "march"),
    ("My mother is a teacher.", "What does my mom do?", "teacher"),
    ("I vacation in Hawaii every summer.", "Where do I vacation?", "hawaii"),
    ("My favorite band is The Beatles.", "What band do I like?", "beatles"),
    ("I drive to work every day.", "How do I commute?", "drive"),
    ("My car is blue.", "What color is my car?", "blue"),
    ("I have two brothers.", "How many siblings?", "two"),
    ("I speak French fluently.", "What language do I speak?", "french"),
    ("I studied physics in college.", "What did I study?", "physics"),
]


class StressTestRetrievalExperiment(BaseExperiment):
    """
    Extended stress test for retrieval degradation with extreme distractor counts.
    
    Tests retrieval at 200, 500, 1000 distractors to:
    1. Break the 100% ceiling effect
    2. Measure degradation curve shape (OU vs Brownian)
    3. Show exponential decay to baseline as predicted by OU process
    """
    
    experiment_name = "stress_test_retrieval"
    
    def run(self) -> ExperimentResult:
        """Execute the stress test retrieval experiment."""
        self.log("Starting STRESS TEST retrieval experiment...")
        
        # Stress test with increasing distractor counts
        # Includes extreme values (10k, 100k) to absolutely break any ceiling effect
        distractor_counts = [50, 100, 200, 500, 1000, 10000, 100000]
        num_trials = 5  # Increased from 2 to 5 for statistical rigor
        num_facts = 20
        
        self.log(f"Testing with {num_facts} facts across distractor counts: {distractor_counts}")
        self.log(f"Running {num_trials} trials per configuration")
        
        # Results storage
        all_results = []
        
        for trial_idx in range(num_trials):
            self.log(f"\n=== Trial {trial_idx + 1}/{num_trials} ===")
            trial_results = self._run_single_trial(
                num_facts=num_facts,
                distractor_counts=distractor_counts,
                trial_idx=trial_idx
            )
            all_results.append(trial_results)
        
        # Aggregate results across trials
        return self._compile_aggregated_results(all_results, distractor_counts, num_facts, num_trials)
    
    def _run_single_trial(
        self, 
        num_facts: int, 
        distractor_counts: List[int],
        trial_idx: int
    ) -> Dict:
        """Run a single trial across all distractor counts."""
        facts_data = EXTENDED_FACTS[:num_facts]
        
        # Compute embeddings for facts (once per trial)
        self.log(f"Computing embeddings for {num_facts} facts...")
        fact_embeddings = []
        for fact, query, keyword in facts_data:
            emb = self.embedding_client.get_embedding(fact)
            fact_embeddings.append(np.array(emb.embedding))
        
        trial_results = {
            'distractor_counts': [],
            'accuracies': [],
            'mean_ranks': [],
            'median_ranks': [],
            'mean_scores': [],
            'mrr_scores': [],  # Mean Reciprocal Rank - more sensitive metric
        }
        
        for num_distractors in distractor_counts:
            self.log(f"  Testing with {num_distractors} distractors...")
            
            # Generate SEMANTICALLY SIMILAR distractors (harder test)
            # Use personal facts about OTHER people to confuse retrieval
            distractor_embeddings = []
            for i in range(num_distractors):
                # Generate semantically similar personal facts (confusers)
                names = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Avery"]
                cities = ["Portland", "Austin", "Denver", "Phoenix", "Miami", "Atlanta", "Chicago"]
                colors = ["red", "green", "purple", "yellow", "orange", "pink"]
                foods = ["pasta", "burger", "sushi", "tacos", "salad"]
                pets = ["Luna", "Charlie", "Bella", "Cooper", "Daisy"]
                companies = ["Microsoft", "Apple", "Amazon", "Meta", "Tesla"]
                universities = ["MIT", "Harvard", "Berkeley", "Yale", "Princeton"]
                instruments = ["piano", "violin", "drums", "flute"]
                movies = ["Interstellar", "Avatar", "Titanic", "Matrix"]
                
                # Generate facts similar to our target facts
                distractor_templates = [
                    f"{np.random.choice(names)}'s favorite color is {np.random.choice(colors)}.",
                    f"{np.random.choice(names)} was born in {np.random.choice(cities)}.",
                    f"{np.random.choice(names)}'s dog is named {np.random.choice(pets)}.",
                    f"{np.random.choice(names)} works at {np.random.choice(companies)}.",
                    f"{np.random.choice(names)} loves {np.random.choice(foods)}.",
                    f"{np.random.choice(names)} lives in {np.random.choice(cities)}.",
                    f"{np.random.choice(names)} graduated from {np.random.choice(universities)}.",
                    f"{np.random.choice(names)} plays {np.random.choice(instruments)}.",
                    f"{np.random.choice(names)}'s favorite movie is {np.random.choice(movies)}.",
                ]
                
                distractor_text = np.random.choice(distractor_templates)
                emb = self.embedding_client.get_embedding(distractor_text)
                distractor_embeddings.append(np.array(emb.embedding))
            
            # Test retrieval
            all_embeddings = fact_embeddings + distractor_embeddings
            all_embeddings_matrix = np.array(all_embeddings)
            
            retrieval_scores = []
            retrieval_ranks = []
            correct_count = 0
            
            for i, (fact, query, keyword) in enumerate(facts_data):
                # Get query embedding
                query_emb = np.array(self.embedding_client.get_embedding(query).embedding)
                
                # Compute cosine similarities
                similarities = np.dot(all_embeddings_matrix, query_emb)
                similarities = similarities / (
                    np.linalg.norm(all_embeddings_matrix, axis=1) * np.linalg.norm(query_emb)
                )
                
                # Find rank of correct fact
                sorted_indices = np.argsort(similarities)[::-1]
                rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed
                
                # Accuracy: 1 if rank <= 5, else 0
                if rank <= 5:
                    correct_count += 1
                    score = 1.0 / rank
                else:
                    score = 0.0
                
                retrieval_scores.append(score)
                retrieval_ranks.append(int(rank))
            
            accuracy = correct_count / num_facts
            mean_rank = np.mean(retrieval_ranks)
            median_rank = np.median(retrieval_ranks)
            mean_score = np.mean(retrieval_scores)
            
            # Calculate Mean Reciprocal Rank (MRR) - more sensitive metric
            mrr = np.mean([1.0 / r for r in retrieval_ranks])
            
            trial_results['distractor_counts'].append(num_distractors)
            trial_results['accuracies'].append(accuracy)
            trial_results['mean_ranks'].append(mean_rank)
            trial_results['median_ranks'].append(median_rank)
            trial_results['mean_scores'].append(mean_score)
            trial_results['mrr_scores'].append(mrr)
            
            self.log(f"    Accuracy: {accuracy:.1%}, Mean Rank: {mean_rank:.1f}, "
                    f"Median Rank: {median_rank:.0f}, MRR: {mrr:.4f}")
        
        return trial_results
    
    def _compile_aggregated_results(
        self, 
        all_results: List[Dict],
        distractor_counts: List[int],
        num_facts: int,
        num_trials: int
    ) -> ExperimentResult:
        """Aggregate results across all trials."""
        self.log(f"\n=== Aggregating results across {num_trials} trials ===")
        
        # Aggregate by distractor count
        aggregated = {dc: {'accuracies': [], 'ranks': [], 'median_ranks': [], 'scores': [], 'mrr': []}
                     for dc in distractor_counts}
        
        for trial_result in all_results:
            for i, dc in enumerate(trial_result['distractor_counts']):
                aggregated[dc]['accuracies'].append(trial_result['accuracies'][i])
                aggregated[dc]['ranks'].append(trial_result['mean_ranks'][i])
                aggregated[dc]['median_ranks'].append(trial_result['median_ranks'][i])
                aggregated[dc]['scores'].append(trial_result['mean_scores'][i])
                aggregated[dc]['mrr'].append(trial_result['mrr_scores'][i])
        
        # Compute statistics
        mean_accuracies = []
        std_accuracies = []
        mean_ranks = []
        median_ranks = []
        mean_scores = []
        std_scores = []
        mean_mrr = []
        std_mrr = []
        
        for dc in distractor_counts:
            mean_acc = np.mean(aggregated[dc]['accuracies'])
            std_acc = np.std(aggregated[dc]['accuracies'])
            mean_rank = np.mean(aggregated[dc]['ranks'])
            median_rank = np.mean(aggregated[dc]['median_ranks'])
            mean_score = np.mean(aggregated[dc]['scores'])
            std_score = np.std(aggregated[dc]['scores'])
            mrr = np.mean(aggregated[dc]['mrr'])
            std_mrr_val = np.std(aggregated[dc]['mrr'])
            
            mean_accuracies.append(mean_acc)
            std_accuracies.append(std_acc)
            mean_ranks.append(mean_rank)
            median_ranks.append(median_rank)
            mean_scores.append(mean_score)
            std_scores.append(std_score)
            mean_mrr.append(mrr)
            std_mrr.append(std_mrr_val)
            
            self.log(f"  {dc} distractors: Accuracy={mean_acc:.1%} ±{std_acc:.1%}, "
                    f"MRR={mrr:.4f} ±{std_mrr_val:.4f}, Mean Rank={mean_rank:.2f}")
        
        # Fit degradation curves
        distractor_array = np.array(distractor_counts)
        accuracy_array = np.array(mean_accuracies)
        mrr_array = np.array(mean_mrr)
        
        # Try OU-style exponential decay on BOTH accuracy and MRR
        accuracy_fit = self._fit_ou_decay(distractor_array, accuracy_array)
        mrr_fit = self._fit_ou_decay(distractor_array, mrr_array)
        
        # Determine which metric shows better degradation
        # If accuracy stays >95% but MRR degrades, use MRR as primary metric
        ceiling_effect = mean_accuracies[0] > 0.95 and mean_accuracies[-1] > 0.95
        primary_metric = "mrr" if ceiling_effect else "accuracy"
        
        if ceiling_effect:
            self.log("\n⚠️  CEILING EFFECT DETECTED: Accuracy stays >95%")
            self.log("    Using MRR as primary degradation metric (more sensitive)")
            primary_fit = mrr_fit
        else:
            self.log("\n✓ Accuracy degradation detected, using as primary metric")
            primary_fit = accuracy_fit
        
        # Also fit power law for comparison
        power_law_fit = None
        try:
            if np.all(accuracy_array > 0):
                pl_fit = fit_power_law(distractor_array, accuracy_array, with_offset=False)
                power_law_fit = {
                    "amplitude": float(pl_fit.amplitude),
                    "exponent": float(pl_fit.exponent),
                    "r_squared": float(pl_fit.r_squared)
                }
        except Exception as e:
            self.log(f"Power law fit failed: {e}")
        
        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "num_trials": num_trials,
                "num_facts": num_facts,
                "degradation_detected": mean_accuracies[0] > mean_accuracies[-1] or mean_mrr[0] > mean_mrr[-1],
                "ceiling_broken": mean_accuracies[0] < 0.99,  # Did we break 100% ceiling?
                "ceiling_effect": ceiling_effect,  # Is accuracy staying >95%?
                "primary_metric": primary_metric,  # "accuracy" or "mrr"
                # Primary metric fit (accuracy or MRR depending on ceiling effect)
                "ou_decay_rate": primary_fit['decay_rate'] if primary_fit else None,
                "ou_baseline": primary_fit['baseline'] if primary_fit else None,
                "ou_r_squared": primary_fit['r_squared'] if primary_fit else None,
                # Separate accuracy and MRR fits
                "accuracy_ou_decay_rate": accuracy_fit['decay_rate'] if accuracy_fit else None,
                "accuracy_ou_r_squared": accuracy_fit['r_squared'] if accuracy_fit else None,
                "mrr_ou_decay_rate": mrr_fit['decay_rate'] if mrr_fit else None,
                "mrr_ou_baseline": mrr_fit['baseline'] if mrr_fit else None,
                "mrr_ou_r_squared": mrr_fit['r_squared'] if mrr_fit else None,
                # MRR degradation percentage
                "mrr_degradation_pct": float((mean_mrr[0] - mean_mrr[-1]) / mean_mrr[0] * 100) if mean_mrr[0] > 0 else 0,
            },
            data={
                "distractor_counts": distractor_counts,
                "mean_accuracies": mean_accuracies,
                "std_accuracies": std_accuracies,
                "mean_ranks": mean_ranks,
                "median_ranks": median_ranks,
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "mean_mrr": mean_mrr,
                "std_mrr": std_mrr,
                "accuracy_fit": accuracy_fit,
                "mrr_fit": mrr_fit,
                "power_law_fit": power_law_fit,
                "all_trial_results": all_results,
            }
        )
        return self.result
    
    def _fit_ou_decay(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict]:
        """
        Fit OU process-style exponential decay: y = A * exp(-k*x) + baseline.
        
        This tests the COFFEE Law prediction that variance saturates rather than
        growing indefinitely (Brownian motion).
        """
        try:
            from scipy.optimize import curve_fit
            
            def ou_model(x, A, k, baseline):
                return A * np.exp(-k * x) + baseline
            
            # Initial guess: starts at max, decays to min
            p0 = [y[0] - y[-1], 0.001, y[-1]]
            
            popt, pcov = curve_fit(ou_model, x, y, p0=p0, maxfev=10000)
            
            A, k, baseline = popt
            y_pred = ou_model(x, A, k, baseline)
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.log(f"OU Decay Fit: A={A:.3f}, k={k:.6f}, baseline={baseline:.3f}, R²={r_squared:.4f}")
            
            return {
                "amplitude": float(A),
                "decay_rate": float(k),
                "baseline": float(baseline),
                "r_squared": float(r_squared),
                "model": "exponential_decay"
            }
        except Exception as e:
            self.log(f"OU decay fit failed: {e}")
            return None
    
    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot stress test results showing degradation curve."""
        if not self.result or not self.result.success:
            self.log("No results to plot")
            return
        
        data = self.result.data
        metrics = self.result.metrics
        
        distractor_counts = np.array(data["distractor_counts"])
        mean_accuracies = np.array(data["mean_accuracies"])
        std_accuracies = np.array(data["std_accuracies"])
        mean_mrr = np.array(data["mean_mrr"])
        std_mrr = np.array(data["std_mrr"])
        
        accuracy_fit = data.get("accuracy_fit")
        mrr_fit = data.get("mrr_fit")
        ceiling_effect = metrics.get("ceiling_effect", False)
        primary_metric = metrics.get("primary_metric", "accuracy")
        
        # Create 2x2 subplot if ceiling effect, otherwise 1x2
        if ceiling_effect:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy vs Distractors
        ax1.errorbar(distractor_counts, mean_accuracies, yerr=std_accuracies,
                    fmt='o-', markersize=8, capsize=5, capthick=2,
                    color='steelblue', ecolor='gray', label='Observed (±1σ)')
        
        # OU fit line for accuracy
        if accuracy_fit:
            x_smooth = np.linspace(distractor_counts[0], distractor_counts[-1], 200)
            y_smooth = accuracy_fit['amplitude'] * np.exp(-accuracy_fit['decay_rate'] * x_smooth) + accuracy_fit['baseline']
            ax1.plot(x_smooth, y_smooth, 'r--', linewidth=2,
                    label=f"OU Fit: exp(-{accuracy_fit['decay_rate']:.4f}·x) (R²={accuracy_fit['r_squared']:.3f})")
        
        ax1.set_xlabel('Number of Distractors', fontsize=12)
        ax1.set_ylabel('Retrieval Accuracy (Top-5)', fontsize=12)
        title_suffix = " [CEILING EFFECT]" if ceiling_effect else ""
        ax1.set_title(f'Accuracy Degradation{title_suffix}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Annotation
        ceiling_broken = "✓ Ceiling Broken" if metrics['ceiling_broken'] else "✗ Still at Ceiling"
        ax1.text(0.02, 0.98, f"Trials: {metrics['num_trials']}\n{ceiling_broken}",
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Plot 2: MRR vs Distractors (more sensitive metric)
        ax2.errorbar(distractor_counts, mean_mrr, yerr=std_mrr,
                    fmt='o-', markersize=8, capsize=5, capthick=2,
                    color='darkviolet', ecolor='gray', label='MRR (±1σ)')
        
        # OU fit line for MRR
        if mrr_fit:
            x_smooth = np.linspace(distractor_counts[0], distractor_counts[-1], 200)
            y_smooth = mrr_fit['amplitude'] * np.exp(-mrr_fit['decay_rate'] * x_smooth) + mrr_fit['baseline']
            ax2.plot(x_smooth, y_smooth, 'r--', linewidth=2,
                    label=f"OU Fit: exp(-{mrr_fit['decay_rate']:.4f}·x) (R²={mrr_fit['r_squared']:.3f})")
        
        ax2.set_xlabel('Number of Distractors', fontsize=12)
        ax2.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12)
        title_suffix = " [PRIMARY METRIC]" if ceiling_effect else " [SENSITIVE METRIC]"
        ax2.set_title(f'MRR Degradation{title_suffix}', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Annotation showing MRR degradation
        mrr_deg = metrics.get('mrr_degradation_pct', 0)
        ax2.text(0.02, 0.98, f"MRR Degradation: {mrr_deg:.1f}%",
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Additional plots if ceiling effect detected
        if ceiling_effect:
            # Plot 3: Mean Rank vs Distractors
            mean_ranks = np.array(data["mean_ranks"])
            ax3.plot(distractor_counts, mean_ranks, 'o-', markersize=8,
                    color='darkgreen', linewidth=2, label='Mean Rank')
            ax3.set_xlabel('Number of Distractors', fontsize=12)
            ax3.set_ylabel('Mean Retrieval Rank', fontsize=12)
            ax3.set_title('Mean Rank Degradation', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3, which='both')
            
            # Plot 4: Comparison of degradation rates
            # Normalize both to [0, 1] for comparison
            norm_acc = (mean_accuracies - mean_accuracies.min()) / (mean_accuracies.max() - mean_accuracies.min() + 1e-10)
            norm_mrr = (mean_mrr - mean_mrr.min()) / (mean_mrr.max() - mean_mrr.min() + 1e-10)
            
            ax4.plot(distractor_counts, 1 - norm_acc, 'o-', linewidth=2, label='Accuracy Degradation (normalized)', color='steelblue')
            ax4.plot(distractor_counts, 1 - norm_mrr, 's-', linewidth=2, label='MRR Degradation (normalized)', color='darkviolet')
            ax4.set_xlabel('Number of Distractors', fontsize=12)
            ax4.set_ylabel('Normalized Degradation', fontsize=12)
            ax4.set_title('Sensitivity Comparison', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            ax4.text(0.5, 0.95, f"Primary Metric: {primary_metric.upper()}\n(More sensitive to degradation)",
                    transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                    ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
