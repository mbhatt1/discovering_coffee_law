"""
Lost in the Middle Experiment - Liu et al. (2023) Protocol Implementation.

This experiment implements the proper "Lost in the Middle" protocol from
Liu et al. (2023) for the COFFEE Law validation paper.

The protocol:
1. Create a context with N documents (e.g., 20 documents)
2. One document contains the answer to a question
3. Vary the position of the relevant document across:
   - Beginning, 10%, 25%, 50%, 75%, 90%, End
4. Ask the model to answer the question
5. Measure accuracy at each position

Expected findings:
- U-shaped accuracy curve (high at beginning/end, low in middle)
- The "middle" degradation corresponds to where OU variance saturates
- Correlation between extraction accuracy and variance saturation curve

Reference:
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023).
Lost in the Middle: How Language Models Use Long Contexts.
"""

import time
import random
import numpy as np
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import stats

from .base import BaseExperiment, ExperimentResult


# Diverse question-answer pairs to avoid memorization effects
# Format: (target_fact, question, expected_answer_keywords)
QA_PAIRS = [
    (
        "The capital of Verdania is Crystallis, a city known for its glass architecture.",
        "What is the capital of Verdania?",
        ["crystallis"]
    ),
    (
        "Dr. Helena Marsh discovered the element Aurinium in 1987 at Stanford Laboratory.",
        "Who discovered Aurinium and when?",
        ["helena marsh", "1987"]
    ),
    (
        "The Thornwood Protocol requires exactly 7 signatures from senior board members.",
        "How many signatures does the Thornwood Protocol require?",
        ["7", "seven"]
    ),
    (
        "Project Nightingale was funded with a budget of $4.3 million from the defense department.",
        "What was the budget for Project Nightingale?",
        ["4.3 million", "4.3m", "$4,300,000"]
    ),
    (
        "The Riverside Museum opens at 9:30 AM and closes at 6:00 PM on Tuesdays.",
        "What are the Riverside Museum's Tuesday hours?",
        ["9:30", "6:00", "6 pm"]
    ),
    (
        "Agent codename 'Falcon' was assigned to the Berlin operation in sector 12.",
        "Which sector was Falcon assigned to in Berlin?",
        ["12", "sector 12"]
    ),
    (
        "The Velarian Treaty was signed by exactly 14 nations on March 15, 2019.",
        "How many nations signed the Velarian Treaty?",
        ["14", "fourteen"]
    ),
    (
        "CEO Miranda Chen announced quarterly profits of $892 million at the shareholder meeting.",
        "What were the quarterly profits announced by Miranda Chen?",
        ["892 million", "$892"]
    ),
    (
        "The optimal reaction temperature for synthesizing compound XR-47 is 127 degrees Celsius.",
        "What temperature is needed to synthesize compound XR-47?",
        ["127", "127 degrees"]
    ),
    (
        "The Blackwood Estate spans 2,340 acres and includes three natural lakes.",
        "How many acres is the Blackwood Estate?",
        ["2,340", "2340"]
    ),
    (
        "Flight KL-4872 departs from Terminal B at 14:25 on weekdays.",
        "What terminal does flight KL-4872 depart from?",
        ["terminal b", "b"]
    ),
    (
        "Professor Alan Whitmore received the prestigious Nexus Award for his work on quantum coherence.",
        "What award did Professor Alan Whitmore receive?",
        ["nexus award", "nexus"]
    ),
    (
        "The ancient city of Kareth was founded in 847 BCE by the Miran civilization.",
        "When was the city of Kareth founded?",
        ["847 bce", "847"]
    ),
    (
        "The primary ingredient in Stellaris Serum is a rare compound extracted from Arctic moss.",
        "What is the primary ingredient in Stellaris Serum?",
        ["arctic moss", "compound"]
    ),
    (
        "Director James Holloway approved the $15.7 million renovation of Building C.",
        "How much was approved for the renovation of Building C?",
        ["15.7 million", "$15.7"]
    ),
]

# Distractor templates for generating diverse filler documents
DISTRACTOR_TEMPLATES = [
    "The {organization} reported a {percent}% increase in {metric} for the {time_period}.",
    "According to {researcher}, the study found that {topic} has significant implications for {field}.",
    "The {building} in {city} was designed by architect {name} in {year}.",
    "Local authorities announced that the {event} will take place at {location} on {date}.",
    "The {company} merged with {other_company} in a deal valued at {amount}.",
    "Recent findings suggest that {phenomenon} may be linked to {factor}.",
    "The {committee} voted {vote_count} in favor of the {proposal}.",
    "Weather forecasts indicate {weather_type} conditions expected in {region} this {season}.",
    "{title} {name} stated that the {project} is expected to complete by {deadline}.",
    "Historical records show that {artifact} was discovered in {location} during the {era}.",
    "The {department} released its annual report showing {finding} in {category}.",
    "Analysts predict that {market} will see {change} over the next {timeframe}.",
    "The {festival} attracted {attendance} visitors this year, a record for the {venue}.",
    "Research indicates that {species} populations have {trend} by {percentage} since {baseline_year}.",
    "The {publication} featured an article about {subject} written by {author}.",
]

# Fill-in values for distractor templates
FILL_VALUES = {
    "organization": ["Global Analytics Corp", "Eastern Research Institute", "Pacific Development Group",
                    "Northern Technology Alliance", "Southern Commerce Bureau", "Western Innovation Lab"],
    "percent": ["12", "27", "5", "43", "8", "31", "19", "56"],
    "metric": ["productivity", "engagement", "efficiency", "revenue", "participation", "satisfaction"],
    "time_period": ["Q3 2024", "fiscal year", "past quarter", "last month", "annual review"],
    "researcher": ["Dr. Emily Watson", "Prof. Richard Lee", "Dr. Sarah Martinez", "Prof. John Kim"],
    "topic": ["climate adaptation", "neural plasticity", "market dynamics", "social behavior"],
    "field": ["public policy", "medical research", "economic planning", "environmental science"],
    "building": ["Central Tower", "Heritage Hall", "Innovation Center", "Civic Complex"],
    "city": ["Melbourne", "Toronto", "Munich", "Singapore", "Oslo", "Denver"],
    "name": ["Marcus Webb", "Elena Sokolov", "David Chen", "Maria Santos", "Thomas Berg"],
    "year": ["1952", "1978", "1994", "2003", "2011", "2018"],
    "event": ["annual conference", "trade exhibition", "community festival", "symposium"],
    "location": ["Convention Center", "Grand Plaza", "Heritage Park", "City Hall"],
    "date": ["March 15", "July 22", "September 8", "November 30", "April 3"],
    "company": ["Nexus Industries", "Vertex Solutions", "Pinnacle Corp", "Summit Enterprises"],
    "other_company": ["Atlas Tech", "Nova Systems", "Horizon Group", "Eclipse Partners"],
    "amount": ["$2.1 billion", "$450 million", "$780 million", "$3.4 billion"],
    "phenomenon": ["atmospheric variation", "behavioral patterns", "market fluctuations"],
    "factor": ["environmental conditions", "regulatory changes", "demographic shifts"],
    "committee": ["Advisory Board", "Planning Commission", "Review Panel", "Ethics Council"],
    "vote_count": ["7-2", "12-3", "unanimous", "8-4", "5-2"],
    "proposal": ["new guidelines", "budget amendment", "policy revision", "strategic plan"],
    "weather_type": ["mild", "severe", "variable", "stable", "extreme"],
    "region": ["coastal areas", "mountain regions", "central districts", "northern territories"],
    "season": ["spring", "summer", "autumn", "winter", "weekend"],
    "title": ["Director", "Chairman", "Secretary", "Commissioner", "Coordinator"],
    "project": ["infrastructure upgrade", "modernization effort", "expansion plan"],
    "deadline": ["Q4 2025", "early 2026", "late 2024", "mid-2025"],
    "artifact": ["ancient pottery", "bronze implements", "stone carvings", "textile fragments"],
    "era": ["Bronze Age", "Medieval period", "Victorian era", "Early Modern period"],
    "department": ["Planning Ministry", "Commerce Bureau", "Health Authority", "Transport Division"],
    "finding": ["steady growth", "significant improvements", "notable challenges"],
    "category": ["public services", "infrastructure", "healthcare delivery", "education"],
    "market": ["tech sector", "commodities", "real estate", "emerging markets"],
    "change": ["moderate growth", "significant volatility", "steady decline", "rapid expansion"],
    "timeframe": ["two quarters", "fiscal year", "18 months", "five years"],
    "festival": ["Arts Festival", "Heritage Days", "Innovation Week", "Cultural Celebration"],
    "attendance": ["45,000", "120,000", "78,000", "250,000"],
    "venue": ["city center", "exhibition grounds", "waterfront area", "historic district"],
    "species": ["migratory bird", "marine mammal", "forest ecosystem", "pollinator"],
    "trend": ["increased", "decreased", "stabilized", "fluctuated"],
    "percentage": ["15%", "32%", "8%", "47%", "23%"],
    "baseline_year": ["2010", "2000", "2015", "1990"],
    "publication": ["Science Today", "Global Affairs Review", "Tech Quarterly", "Nature Journal"],
    "subject": ["renewable energy", "urban development", "genetic research", "space exploration"],
    "author": ["J. Morrison", "L. Nakamura", "R. Okonkwo", "S. Petrov"],
}


class LostInMiddleExperiment(BaseExperiment):
    """
    Implements the "Lost in the Middle" protocol from Liu et al. (2023).

    This experiment validates that:
    1. Extraction accuracy follows a U-shaped curve across positions
    2. Middle positions show degraded accuracy
    3. The accuracy curve correlates with OU variance saturation dynamics

    The COFFEE Law predicts that the "lost in the middle" phenomenon
    corresponds to the regime where variance saturates (OU process),
    and attention becomes uniformly distributed rather than focused.
    """

    experiment_name = "lost_in_middle"

    # Position labels for Liu et al. protocol
    POSITION_LABELS = ["beginning", "10%", "25%", "50%", "75%", "90%", "end"]
    POSITION_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    def __init__(
        self,
        config,
        client=None,
        embedding_client=None,
        quick_mode: bool = False,
        stress_test: bool = False,
    ):
        """
        Initialize the Lost in the Middle experiment.

        Args:
            config: Experiment configuration
            client: OpenAI client
            embedding_client: Embedding client
            quick_mode: If True, use reduced parameters for faster testing
            stress_test: If True, use 10x harder conditions to reveal U-curve
        """
        super().__init__(config, client, embedding_client)
        self.quick_mode = quick_mode
        self.stress_test = stress_test

    def run(self) -> ExperimentResult:
        """Execute the Lost in the Middle experiment."""
        self.log("Starting LOST IN THE MIDDLE experiment (Liu et al. 2023 protocol)...")

        # Configuration - 10x harder for stress test
        if self.stress_test:
            num_documents = 200  # 10x more documents
            num_trials = 5       # More trials for statistical power
            num_qa_pairs = len(QA_PAIRS)  # All QA pairs
            docs_per_distractor = 3  # Make distractors multi-sentence
            self.log("*** STRESS TEST MODE: 10x harder conditions ***")
        elif self.quick_mode:
            num_documents = 10
            num_trials = 1
            num_qa_pairs = 3
            docs_per_distractor = 1
        else:
            num_documents = 20
            num_trials = 3
            num_qa_pairs = min(10, len(QA_PAIRS))
            docs_per_distractor = 1

        self.docs_per_distractor = docs_per_distractor

        self.log(f"Configuration: {num_documents} documents, {num_trials} trials, {num_qa_pairs} QA pairs")
        self.log(f"Testing positions: {self.POSITION_LABELS}")

        # Storage for results
        all_results = {pos: [] for pos in self.POSITION_LABELS}
        detailed_results = []

        # Select QA pairs
        selected_qa = random.sample(QA_PAIRS, num_qa_pairs)

        for qa_idx, (target_fact, question, expected_keywords) in enumerate(selected_qa):
            self.log(f"\n=== QA Pair {qa_idx + 1}/{num_qa_pairs} ===")
            self.log(f"Question: {question}")

            for trial_idx in range(num_trials):
                self.log(f"  Trial {trial_idx + 1}/{num_trials}")

                # Generate distractor documents for this trial
                distractors = self._create_document_set(num_documents - 1)

                for pos_idx, (pos_label, pos_frac) in enumerate(
                    zip(self.POSITION_LABELS, self.POSITION_FRACTIONS)
                ):
                    # Place target at specified position
                    context_docs = self._place_target_at_position(
                        distractors=distractors,
                        target_doc=target_fact,
                        position_fraction=pos_frac,
                        total_docs=num_documents
                    )

                    # Create full context
                    context = self._format_context(context_docs)

                    # Evaluate extraction
                    is_correct, response = self._evaluate_extraction(
                        context=context,
                        question=question,
                        expected_keywords=expected_keywords
                    )

                    # Store result
                    all_results[pos_label].append(is_correct)
                    detailed_results.append({
                        'qa_idx': qa_idx,
                        'trial_idx': trial_idx,
                        'position': pos_label,
                        'position_fraction': pos_frac,
                        'correct': is_correct,
                        'question': question,
                        'response': response,
                    })

                    if pos_idx % 2 == 0:
                        self.log(f"    Position {pos_label}: {'correct' if is_correct else 'incorrect'}")

        # Compute accuracy curve
        accuracy_by_position = self._compute_accuracy_curve(all_results)

        # Analyze U-curve and correlation with OU dynamics
        analysis = self._analyze_results(accuracy_by_position, detailed_results)

        self.result = ExperimentResult(
            experiment_name=self.experiment_name,
            success=True,
            metrics={
                "num_documents": num_documents,
                "num_trials": num_trials,
                "num_qa_pairs": num_qa_pairs,
                "total_evaluations": len(detailed_results),
                # Accuracy metrics
                "accuracy_beginning": accuracy_by_position["beginning"]["mean"],
                "accuracy_middle": accuracy_by_position["50%"]["mean"],
                "accuracy_end": accuracy_by_position["end"]["mean"],
                "overall_accuracy": analysis["overall_accuracy"],
                # U-curve metrics
                "u_curve_detected": analysis["u_curve_detected"],
                "middle_degradation": analysis["middle_degradation"],
                "u_curve_depth": analysis["u_curve_depth"],
                # OU correlation metrics
                "position_variance_correlation": analysis["position_variance_correlation"],
                "min_accuracy_position": analysis["min_accuracy_position"],
            },
            data={
                "position_labels": self.POSITION_LABELS,
                "position_fractions": self.POSITION_FRACTIONS,
                "accuracy_by_position": accuracy_by_position,
                "detailed_results": detailed_results,
                "analysis": analysis,
            }
        )

        return self.result

    def _create_document_set(self, num_distractors: int) -> List[str]:
        """
        Generate a set of distractor documents.

        Args:
            num_distractors: Number of distractor documents to generate

        Returns:
            List of distractor document strings
        """
        distractors = []
        docs_per = getattr(self, 'docs_per_distractor', 1)

        for _ in range(num_distractors):
            # Generate multiple sentences per distractor for stress test
            sentences = []
            for _ in range(docs_per):
                template = random.choice(DISTRACTOR_TEMPLATES)

                # Fill in template with random values
                doc = template
                for key, values in FILL_VALUES.items():
                    placeholder = "{" + key + "}"
                    if placeholder in doc:
                        doc = doc.replace(placeholder, random.choice(values), 1)

                sentences.append(doc)

            distractors.append(" ".join(sentences))

        return distractors

    def _place_target_at_position(
        self,
        distractors: List[str],
        target_doc: str,
        position_fraction: float,
        total_docs: int
    ) -> List[str]:
        """
        Insert target document at specified relative position.

        Args:
            distractors: List of distractor documents
            target_doc: The target document containing the answer
            position_fraction: Position as fraction (0.0 = beginning, 1.0 = end)
            total_docs: Total number of documents

        Returns:
            List of documents with target at specified position
        """
        # Calculate target position index
        if position_fraction == 0.0:
            target_idx = 0
        elif position_fraction == 1.0:
            target_idx = total_docs - 1
        else:
            target_idx = int(position_fraction * (total_docs - 1))

        # Ensure we have enough distractors
        distractors_copy = distractors.copy()
        while len(distractors_copy) < total_docs - 1:
            distractors_copy.extend(distractors[:total_docs - 1 - len(distractors_copy)])

        distractors_copy = distractors_copy[:total_docs - 1]

        # Shuffle distractors
        random.shuffle(distractors_copy)

        # Insert target at specified position
        documents = distractors_copy[:target_idx] + [target_doc] + distractors_copy[target_idx:]

        return documents

    def _format_context(self, documents: List[str]) -> str:
        """
        Format documents into a context string.

        Args:
            documents: List of documents

        Returns:
            Formatted context string with document markers
        """
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}: {doc}")

        return "\n\n".join(formatted_docs)

    def _evaluate_extraction(
        self,
        context: str,
        question: str,
        expected_keywords: List[str]
    ) -> Tuple[bool, str]:
        """
        Evaluate if the model correctly extracts the answer.

        Args:
            context: The document context
            question: The question to answer
            expected_keywords: Keywords that should appear in correct answer

        Returns:
            (is_correct, response) tuple
        """
        prompt = f"""You are given a set of documents. Answer the question based ONLY on the information provided in these documents.

{context}

Question: {question}

Provide a brief, direct answer based only on the documents above. If the answer is not in the documents, say "Not found in documents."

Answer:"""

        try:
            response = self.generate_completion(
                prompt=prompt,
                max_tokens=150,
                temperature=0.0  # Deterministic for evaluation
            )

            answer = response["content"].lower().strip()

            # Check if any expected keyword is in the answer
            is_correct = any(
                keyword.lower() in answer
                for keyword in expected_keywords
            )

            # Also check for "not found" responses as incorrect
            if "not found" in answer or "cannot find" in answer or "no information" in answer:
                is_correct = False

            return is_correct, response["content"]

        except Exception as e:
            self.log(f"    Error during evaluation: {e}")
            return False, f"Error: {str(e)}"

    def _compute_accuracy_curve(
        self,
        all_results: Dict[str, List[bool]]
    ) -> Dict[str, Dict]:
        """
        Compute accuracy statistics by position.

        Args:
            all_results: Dictionary mapping position labels to lists of correct/incorrect

        Returns:
            Dictionary with accuracy statistics per position
        """
        accuracy_by_position = {}

        for pos_label in self.POSITION_LABELS:
            results = all_results[pos_label]
            if results:
                mean_acc = np.mean(results)
                std_acc = np.std(results)
                n = len(results)
                # 95% confidence interval
                ci = 1.96 * std_acc / np.sqrt(n) if n > 0 else 0

                accuracy_by_position[pos_label] = {
                    "mean": float(mean_acc),
                    "std": float(std_acc),
                    "n": n,
                    "ci_95": float(ci),
                    "correct_count": sum(results),
                }

                self.log(f"  {pos_label}: {mean_acc:.1%} ({sum(results)}/{n})")
            else:
                accuracy_by_position[pos_label] = {
                    "mean": 0.0, "std": 0.0, "n": 0, "ci_95": 0.0, "correct_count": 0
                }

        return accuracy_by_position

    def _analyze_results(
        self,
        accuracy_by_position: Dict[str, Dict],
        detailed_results: List[Dict]
    ) -> Dict:
        """
        Analyze results for U-curve and OU correlation.

        Args:
            accuracy_by_position: Accuracy statistics by position
            detailed_results: Detailed results list

        Returns:
            Analysis dictionary with U-curve and correlation metrics
        """
        self.log("\n=== Analyzing Results ===")

        # Extract accuracy values in position order
        accuracies = [accuracy_by_position[pos]["mean"] for pos in self.POSITION_LABELS]
        accuracies = np.array(accuracies)

        # Overall accuracy
        overall_accuracy = np.mean([r["correct"] for r in detailed_results])

        # Check for U-curve: edges should be higher than middle
        edge_accuracy = (accuracies[0] + accuracies[-1]) / 2  # beginning + end
        middle_accuracy = accuracies[3]  # 50% position

        u_curve_detected = edge_accuracy > middle_accuracy
        middle_degradation = edge_accuracy - middle_accuracy

        # U-curve depth: ratio of middle degradation to edge accuracy
        u_curve_depth = middle_degradation / edge_accuracy if edge_accuracy > 0 else 0

        # Find position with minimum accuracy
        min_idx = np.argmin(accuracies)
        min_accuracy_position = self.POSITION_LABELS[min_idx]

        self.log(f"  Edge accuracy: {edge_accuracy:.1%}")
        self.log(f"  Middle accuracy: {middle_accuracy:.1%}")
        self.log(f"  U-curve detected: {u_curve_detected}")
        self.log(f"  Middle degradation: {middle_degradation:.1%}")
        self.log(f"  Min accuracy at: {min_accuracy_position}")

        # Compute correlation between position and accuracy
        # OU model predicts variance saturates in middle -> attention diffuse -> accuracy drops
        # We expect: accuracy inversely correlates with "distance from edges"

        # Distance from nearest edge (0 at edges, max at center)
        edge_distances = np.array([
            min(frac, 1.0 - frac)
            for frac in self.POSITION_FRACTIONS
        ])

        # Spearman correlation between edge distance and accuracy
        # Negative correlation expected (further from edge -> lower accuracy)
        if len(accuracies) > 2:
            spearman_corr, spearman_p = stats.spearmanr(edge_distances, accuracies)
        else:
            spearman_corr, spearman_p = 0.0, 1.0

        self.log(f"  Edge distance vs accuracy correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")

        # Theoretical OU variance saturation comparison
        # In OU process, variance Var(t) = sigma^2 / (2*theta) * (1 - exp(-2*theta*t))
        # This saturates at sigma^2 / (2*theta) as t -> infinity
        # The "middle" of a context can be thought of as where variance has saturated

        # Simulate OU variance saturation curve for comparison
        ou_variance = self._simulate_ou_variance(self.POSITION_FRACTIONS)

        # Accuracy should be high where variance is low (beginning)
        # and low where variance saturates (middle)
        # So: accuracy ~ 1 - normalized_variance
        inverted_ou = 1.0 - ou_variance  # Invert for comparison with accuracy

        if len(accuracies) > 2:
            ou_corr, ou_p = stats.pearsonr(inverted_ou, accuracies)
        else:
            ou_corr, ou_p = 0.0, 1.0

        self.log(f"  Accuracy vs (1 - OU variance) correlation: {ou_corr:.3f} (p={ou_p:.3f})")

        return {
            "overall_accuracy": float(overall_accuracy),
            "edge_accuracy": float(edge_accuracy),
            "middle_accuracy": float(middle_accuracy),
            "u_curve_detected": bool(u_curve_detected),
            "middle_degradation": float(middle_degradation),
            "u_curve_depth": float(u_curve_depth),
            "min_accuracy_position": min_accuracy_position,
            "position_variance_correlation": float(spearman_corr),
            "position_variance_p_value": float(spearman_p),
            "ou_accuracy_correlation": float(ou_corr),
            "ou_accuracy_p_value": float(ou_p),
            "theoretical_ou_variance": ou_variance.tolist(),
            "edge_distances": edge_distances.tolist(),
        }

    def _simulate_ou_variance(self, position_fractions: List[float]) -> np.ndarray:
        """
        Simulate OU variance saturation curve.

        The OU variance formula: Var(t) = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))

        For a context, we model position as "time since start" and also
        account for recency (time to end).

        Args:
            position_fractions: List of position fractions [0, 1]

        Returns:
            Normalized OU variance values
        """
        positions = np.array(position_fractions)

        # OU parameters (normalized for unit interval)
        theta = 3.0  # Mean reversion rate
        sigma = 1.0  # Volatility

        # Variance from start (saturates moving forward)
        var_from_start = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * positions))

        # Variance from end (saturates moving backward)
        var_from_end = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * (1 - positions)))

        # Combined effect: in the middle, both have saturated
        # At edges, one is small (fresh position)
        combined_variance = np.minimum(var_from_start, var_from_end)

        # Alternative: use position-dependent saturation
        # Middle positions are "far" from both edges
        # Model attention as diffusing with distance from fresh position
        edge_freshness = 1.0 - 2 * np.minimum(positions, 1 - positions)  # 0 at edges, 1 in middle
        ou_variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * edge_freshness * 5))

        # Normalize to [0, 1]
        ou_variance = (ou_variance - ou_variance.min()) / (ou_variance.max() - ou_variance.min() + 1e-10)

        return ou_variance

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots for Lost in the Middle results.

        Creates:
        1. Accuracy U-curve by position
        2. Comparison with OU variance saturation curve
        3. Correlation analysis
        """
        if not self.result or not self.result.success:
            self.log("No results to plot")
            return

        data = self.result.data
        metrics = self.result.metrics

        accuracy_by_position = data["accuracy_by_position"]
        analysis = data["analysis"]

        # Extract data for plotting
        positions = self.POSITION_LABELS
        position_fracs = np.array(self.POSITION_FRACTIONS)
        accuracies = np.array([accuracy_by_position[pos]["mean"] for pos in positions])
        ci_95 = np.array([accuracy_by_position[pos]["ci_95"] for pos in positions])

        ou_variance = np.array(analysis["theoretical_ou_variance"])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Accuracy U-curve
        ax1 = axes[0, 0]
        ax1.errorbar(
            position_fracs, accuracies, yerr=ci_95,
            fmt='o-', markersize=10, capsize=5, capthick=2,
            color='steelblue', ecolor='gray', linewidth=2,
            label='Observed Accuracy'
        )

        # Highlight middle region
        ax1.axvspan(0.25, 0.75, alpha=0.1, color='red', label='Middle Region')

        # Mark minimum
        min_idx = np.argmin(accuracies)
        ax1.scatter([position_fracs[min_idx]], [accuracies[min_idx]],
                   s=200, c='red', marker='v', zorder=5, label=f'Min at {positions[min_idx]}')

        ax1.set_xlabel('Relative Position in Context', fontsize=12)
        ax1.set_ylabel('Extraction Accuracy', fontsize=12)
        ax1.set_title('Lost in the Middle: Accuracy by Document Position',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(position_fracs)
        ax1.set_xticklabels(positions, rotation=45, ha='right')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='lower center')
        ax1.grid(True, alpha=0.3)

        # Annotation
        u_curve = "U-curve Detected" if metrics["u_curve_detected"] else "U-curve Not Detected"
        ax1.text(0.02, 0.98, f"{u_curve}\nDepth: {metrics['u_curve_depth']:.1%}",
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Plot 2: OU Variance Comparison
        ax2 = axes[0, 1]

        # Plot accuracy (inverted, normalized)
        acc_normalized = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min() + 1e-10)
        ax2.plot(position_fracs, 1 - acc_normalized, 'o-', linewidth=2, markersize=8,
                color='steelblue', label='1 - Normalized Accuracy')

        # Plot OU variance
        ax2.plot(position_fracs, ou_variance, 's--', linewidth=2, markersize=8,
                color='darkviolet', label='OU Variance (theoretical)')

        ax2.set_xlabel('Relative Position in Context', fontsize=12)
        ax2.set_ylabel('Normalized Value', fontsize=12)
        ax2.set_title('Comparison: Accuracy Loss vs OU Variance Saturation',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(position_fracs)
        ax2.set_xticklabels(positions, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Correlation annotation
        corr = analysis["ou_accuracy_correlation"]
        ax2.text(0.02, 0.98, f"Correlation: {corr:.3f}",
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Plot 3: Bar chart comparison
        ax3 = axes[1, 0]

        x = np.arange(len(positions))
        width = 0.35

        ax3.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
        ax3.bar(x + width/2, 1 - ou_variance, width, label='1 - OU Variance', color='darkviolet', alpha=0.8)

        ax3.set_xlabel('Position', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Accuracy vs (1 - OU Variance) by Position', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(positions, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary text
        summary_lines = [
            "LOST IN THE MIDDLE ANALYSIS",
            "=" * 40,
            "",
            f"Total Evaluations: {metrics['total_evaluations']}",
            f"Documents per Context: {metrics['num_documents']}",
            f"QA Pairs Tested: {metrics['num_qa_pairs']}",
            "",
            "ACCURACY SUMMARY:",
            f"  Beginning: {metrics['accuracy_beginning']:.1%}",
            f"  Middle (50%): {metrics['accuracy_middle']:.1%}",
            f"  End: {metrics['accuracy_end']:.1%}",
            f"  Overall: {metrics['overall_accuracy']:.1%}",
            "",
            "U-CURVE ANALYSIS:",
            f"  U-curve Detected: {'Yes' if metrics['u_curve_detected'] else 'No'}",
            f"  Middle Degradation: {metrics['middle_degradation']:.1%}",
            f"  Curve Depth: {metrics['u_curve_depth']:.1%}",
            f"  Min Accuracy at: {metrics['min_accuracy_position']}",
            "",
            "OU CORRELATION:",
            f"  Position-Variance Corr: {metrics['position_variance_correlation']:.3f}",
            f"  Accuracy-OU Corr: {analysis['ou_accuracy_correlation']:.3f}",
        ]

        summary_text = "\n".join(summary_lines)
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"Plot saved to {save_path}")

        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
