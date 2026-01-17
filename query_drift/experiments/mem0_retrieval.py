"""
Memory Retrieval Experiment - Demonstrates "Lost in the Middle" phenomenon.

Uses embedding-based similarity search to show how retrieval quality
degrades as more content is added between the target and query.
"""

import time
import uuid
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseExperiment, ExperimentResult
from ..utils.math import fit_power_law

# Try to import mem0
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Memory = None


# Facts with clear semantic content for retrieval
FACTS_AND_QUERIES = [
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
]


class Mem0RetrievalExperiment(BaseExperiment):
    """
    Experiment demonstrating memory retrieval degradation.

    Uses embedding similarity to show how adding content between
    stored fact and query point degrades retrieval accuracy.
    """

    experiment_name = "mem0_retrieval"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_id = f"exp_{uuid.uuid4().hex[:8]}"
        self._memory = None

    def run(self) -> ExperimentResult:
        """Execute the retrieval degradation experiment."""
        self.log("Starting memory retrieval experiment...")

        cfg = self.config.mem0_retrieval
        num_memories = min(cfg.num_memories, len(FACTS_AND_QUERIES))

        # Use embedding-based retrieval by default (more reliable and predictable)
        # mem0 integration can be flaky due to its internal processing
        use_embeddings = getattr(cfg, 'use_embeddings', True)

        if not use_embeddings and MEM0_AVAILABLE:
            return self._run_with_mem0(num_memories, cfg)
        else:
            return self._run_with_embeddings(num_memories, cfg)

    def _run_with_embeddings(self, num_memories: int, cfg) -> ExperimentResult:
        """Run using direct embedding comparison (fallback if mem0 unavailable)."""
        self.log("Running with direct embedding search (mem0 not available)")

        facts_data = FACTS_AND_QUERIES[:num_memories]

        # Store embeddings for all facts
        self.log(f"Computing embeddings for {num_memories} facts...")
        fact_embeddings = []
        for fact, query, keyword in facts_data:
            emb = self.embedding_client.get_embedding(fact)
            fact_embeddings.append(np.array(emb.embedding))

        # Add "distractor" embeddings
        num_distractors = num_memories * cfg.distractor_multiplier
        distractor_embeddings = []
        self.log(f"Adding {num_distractors} distractor embeddings...")

        for i in range(num_distractors):
            distractor_text = f"Random fact number {i}: The temperature today is {np.random.randint(50, 100)} degrees and humidity is {np.random.randint(30, 90)} percent."
            emb = self.embedding_client.get_embedding(distractor_text)
            distractor_embeddings.append(np.array(emb.embedding))

        # Now test retrieval - all embeddings in pool
        all_embeddings = fact_embeddings + distractor_embeddings
        all_embeddings_matrix = np.array(all_embeddings)

        self.log("Testing retrieval accuracy...")
        retrieval_scores = []
        retrieval_ranks = []

        for i, (fact, query, keyword) in enumerate(facts_data):
            # Get query embedding
            query_emb = np.array(self.embedding_client.get_embedding(query).embedding)

            # Compute similarities
            similarities = np.dot(all_embeddings_matrix, query_emb)
            similarities = similarities / (np.linalg.norm(all_embeddings_matrix, axis=1) * np.linalg.norm(query_emb))

            # Find rank of correct fact
            sorted_indices = np.argsort(similarities)[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed

            # Score based on rank (1/rank if in top-k, else 0)
            if rank <= cfg.retrieval_limit:
                score = 1.0 / rank
            else:
                score = 0.0

            retrieval_scores.append(score)
            retrieval_ranks.append(int(rank))

            self.log(f"  Fact {i+1}: rank={rank}, score={score:.3f}, sim={similarities[i]:.4f}")

        return self._compile_results(
            num_memories, num_distractors, facts_data,
            retrieval_scores, retrieval_ranks
        )

    def _run_with_mem0(self, num_memories: int, cfg) -> ExperimentResult:
        """Run using mem0 memory system."""
        self.log("Running with mem0 memory system")

        try:
            # Initialize mem0 with simple config (in-memory)
            self._memory = Memory()

            facts_data = FACTS_AND_QUERIES[:num_memories]

            # Store facts
            self.log(f"Storing {num_memories} facts in mem0...")
            for i, (fact, query, keyword) in enumerate(facts_data):
                self._memory.add(fact, user_id=self._user_id)
                time.sleep(cfg.memory_delay_seconds)
                if (i + 1) % 5 == 0:
                    self.log(f"  Stored {i+1}/{num_memories}")

            # Add distractors
            num_distractors = num_memories * cfg.distractor_multiplier
            self.log(f"Adding {num_distractors} distractors...")

            for i in range(num_distractors):
                distractor = f"Weather update {i}: Temperature is {np.random.randint(50, 100)}F with {np.random.randint(30, 90)}% humidity."
                self._memory.add(distractor, user_id=self._user_id)
                time.sleep(cfg.memory_delay_seconds / 2)  # Faster for distractors
                if (i + 1) % 10 == 0:
                    self.log(f"  Added {i+1}/{num_distractors}")

            # Query for each fact
            self.log("Querying for facts...")
            retrieval_scores = []
            retrieval_ranks = []

            for i, (fact, query, keyword) in enumerate(facts_data):
                results = self._memory.search(query, user_id=self._user_id, limit=cfg.retrieval_limit)

                # Check if fact is in results
                found = False
                rank = -1
                keyword_lower = keyword.lower()

                for j, result in enumerate(results):
                    mem_text = ""
                    if isinstance(result, dict):
                        mem_text = result.get('memory', '').lower()
                    elif hasattr(result, 'memory'):
                        mem_text = result.memory.lower()

                    if keyword_lower in mem_text:
                        found = True
                        rank = j + 1
                        break

                if found:
                    score = 1.0 / rank
                else:
                    score = 0.0
                    rank = -1

                retrieval_scores.append(score)
                retrieval_ranks.append(rank)
                self.log(f"  Fact {i+1}: {'Found' if found else 'Not found'} (rank={rank}, score={score:.3f})")

            return self._compile_results(
                num_memories, num_distractors, facts_data,
                retrieval_scores, retrieval_ranks
            )

        except Exception as e:
            self.log(f"mem0 failed: {e}, falling back to embedding search")
            return self._run_with_embeddings(num_memories, cfg)

        finally:
            if self._memory:
                try:
                    self._memory.delete_all(user_id=self._user_id)
                except Exception:
                    pass

    def _compile_results(self, num_memories, num_distractors, facts_data,
                         retrieval_scores, retrieval_ranks) -> ExperimentResult:
        """Compile experiment results."""
        retrieval_scores = np.array(retrieval_scores)
        retrieval_ranks = np.array(retrieval_ranks)
        memory_ages = np.arange(1, num_memories + 1)

        # Metrics
        mean_score = float(np.mean(retrieval_scores))
        found_count = int(np.sum(retrieval_scores > 0))
        retrieval_rate = found_count / num_memories

        self.log(f"\nResults: Mean score = {mean_score:.3f}, Retrieval rate = {retrieval_rate:.1%}")

        # Try to fit power law on non-zero scores
        fit_results = None
        decay_exponent = None

        valid_mask = retrieval_scores > 0
        if np.sum(valid_mask) >= 3:
            try:
                fit = fit_power_law(
                    memory_ages[valid_mask],
                    retrieval_scores[valid_mask],
                    with_offset=False
                )
                fit_results = {
                    "amplitude": float(fit.amplitude),
                    "exponent": float(fit.exponent),
                    "r_squared": float(fit.r_squared)
                }
                decay_exponent = float(fit.exponent)
                self.log(f"Power law fit: score ~ age^{fit.exponent:.3f}, R²={fit.r_squared:.3f}")
            except Exception as e:
                self.log(f"Power law fit failed: {e}")

        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "mean_retrieval_score": mean_score,
                "retrieval_rate": retrieval_rate,
                "found_count": found_count,
                "total_memories": num_memories,
                "num_distractors": num_distractors,
                "decay_exponent": decay_exponent,
                "decay_detected": decay_exponent is not None and decay_exponent < -0.1
            },
            data={
                "memory_ages": memory_ages,
                "retrieval_scores": retrieval_scores,
                "retrieval_ranks": retrieval_ranks,
                "facts": [f[0] for f in facts_data],
                "fit_results": fit_results
            }
        )
        return self.result

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot retrieval score vs memory age."""
        if not self.result or not self.result.success:
            self.log("No results to plot")
            return

        data = self.result.data
        metrics = self.result.metrics

        memory_ages = np.array(data["memory_ages"])
        retrieval_scores = np.array(data["retrieval_scores"])
        fit_results = data.get("fit_results")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(memory_ages, retrieval_scores, s=100, c='steelblue',
                   alpha=0.7, edgecolors='navy', linewidth=1, zorder=3,
                   label='Retrieval Score')

        # Fit line if available
        if fit_results:
            x_fit = np.linspace(1, max(memory_ages), 100)
            y_fit = fit_results["amplitude"] * np.power(x_fit, fit_results["exponent"])
            y_fit = np.maximum(y_fit, 0)
            ax.plot(x_fit, y_fit, 'r-', linewidth=2,
                    label=f'Fit: score ~ age^{fit_results["exponent"]:.2f} (R²={fit_results["r_squared"]:.2f})')

        ax.set_xlabel('Memory Age (older →)', fontsize=12)
        ax.set_ylabel('Retrieval Score (1/rank)', fontsize=12)
        ax.set_title('Memory Retrieval Degradation\n"Lost in the Middle" Effect', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(memory_ages) + 1)
        ax.set_ylim(-0.05, 1.1)

        # Annotation
        text = f"Mean: {metrics['mean_retrieval_score']:.3f}\nRate: {metrics['retrieval_rate']:.0%}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")

        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
