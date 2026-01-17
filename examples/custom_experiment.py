#!/usr/bin/env python3
"""
Custom experiment example showing how to run individual experiments with custom parameters.

This demonstrates fine-grained control over experiment configuration and execution.
"""

from query_drift import ExperimentConfig
from query_drift.config import (
    EmbeddingDriftConfig,
    AlignmentDecayConfig,
    LossScalingConfig,
    Mem0RetrievalConfig,
    ModelConfig,
    OutputConfig,
)
from query_drift.experiments import (
    EmbeddingDriftExperiment,
    AlignmentDecayExperiment,
    LossScalingExperiment,
    Mem0RetrievalExperiment,
)


def run_custom_embedding_drift():
    """Run embedding drift experiment with custom parameters."""
    print("=" * 60)
    print("Custom Embedding Drift Experiment")
    print("=" * 60)

    # Create custom configuration
    config = ExperimentConfig(
        embedding_drift=EmbeddingDriftConfig(
            num_continuations=15,
            max_tokens_drift=80,
            sample_positions=[5, 10, 20, 40, 60, 80],
            temperature=0.9,
        ),
        model=ModelConfig(
            embedding_model="text-embedding-3-small",
            completion_model="gpt-4o-mini",
        ),
        output=OutputConfig(
            output_dir="custom_results",
            save_plots=True,
            verbose=True,
        ),
    )

    # Create and run experiment
    experiment = EmbeddingDriftExperiment(config)
    result = experiment.run(seed_prompt="The theory of quantum mechanics suggests that")

    # Display results
    print(f"\nResults:")
    print(f"  Hurst exponent: {result.exponent:.4f} +/- {result.exponent_std:.4f}")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  Consistent with hypothesis: {result.is_consistent}")

    return result


def run_custom_alignment_decay():
    """Run alignment decay experiment with custom parameters."""
    print("\n" + "=" * 60)
    print("Custom Alignment Decay Experiment")
    print("=" * 60)

    # Create custom configuration
    config = ExperimentConfig(
        alignment_decay=AlignmentDecayConfig(
            num_distractor_rounds=4,
            distractors=[
                "Consider the implications of modern philosophy. ",
                "The economic landscape has shifted significantly. ",
                "Technological advances continue to reshape society. ",
                "Historical precedents inform current debates. ",
                "Cultural factors play a crucial role in outcomes. ",
            ],
        ),
        model=ModelConfig(
            embedding_model="text-embedding-3-small",
        ),
        output=OutputConfig(verbose=True),
    )

    # Create and run experiment
    experiment = AlignmentDecayExperiment(config)
    result = experiment.run(
        key_context="Machine learning models learn patterns from data."
    )

    # Display results
    print(f"\nResults:")
    print(f"  Decay exponent: {result.exponent:.4f} +/- {result.exponent_std:.4f}")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  Consistent with hypothesis: {result.is_consistent}")

    return result


def run_custom_loss_scaling():
    """Run loss scaling experiment with custom parameters."""
    print("\n" + "=" * 60)
    print("Custom Loss Scaling Experiment")
    print("=" * 60)

    # Create custom configuration with specific context lengths
    config = ExperimentConfig(
        loss_scaling=LossScalingConfig(
            context_lengths=[50, 100, 250, 500, 1000, 2000],
            measurements_per_length=4,
            continuation_prompt=" In conclusion,",
        ),
        model=ModelConfig(
            completion_model="gpt-4o-mini",
        ),
        output=OutputConfig(verbose=True),
    )

    # Create and run experiment
    experiment = LossScalingExperiment(config)
    result = experiment.run()

    # Display results
    print(f"\nResults:")
    print(f"  Scaling exponent: {result.exponent:.4f} +/- {result.exponent_std:.4f}")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  Consistent with hypothesis: {result.is_consistent}")

    return result


def run_custom_mem0_retrieval():
    """Run mem0 retrieval experiment with custom parameters."""
    print("\n" + "=" * 60)
    print("Custom mem0 Retrieval Experiment")
    print("=" * 60)

    # Create custom configuration
    config = ExperimentConfig(
        mem0_retrieval=Mem0RetrievalConfig(
            num_memories=15,
            distractor_multiplier=2,
            retrieval_limit=3,
            memory_delay_seconds=0.2,
        ),
        output=OutputConfig(verbose=True),
    )

    # Create and run experiment
    experiment = Mem0RetrievalExperiment(config)
    result = experiment.run()

    # Display results
    print(f"\nResults:")
    print(f"  Decay exponent: {result.exponent:.4f} +/- {result.exponent_std:.4f}")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  Consistent with hypothesis: {result.is_consistent}")

    return result


def run_selected_experiments():
    """Demonstrate running only selected experiments."""
    print("\n" + "=" * 60)
    print("Running Selected Experiments Only")
    print("=" * 60)

    # Create configuration that only runs specific experiments
    config = ExperimentConfig(
        run_embedding_drift=True,
        run_alignment_decay=True,
        run_loss_scaling=False,  # Skip this one
        run_mem0_retrieval=False,  # Skip this one
        embedding_drift=EmbeddingDriftConfig(num_continuations=10),
        alignment_decay=AlignmentDecayConfig(num_distractor_rounds=2),
    )

    # Import validator for batch running
    from query_drift import QueryDriftValidator

    validator = QueryDriftValidator(config)
    results = validator.run_all()

    print("\nCompleted experiments:")
    for name, result in results.items():
        print(f"  {name}: exponent = {result.exponent:.4f}")


def main():
    """Run all custom experiment demonstrations."""
    print("Query Drift Hypothesis - Custom Experiment Examples")
    print("=" * 60)
    print()
    print("This script demonstrates how to run individual experiments")
    print("with custom parameters for fine-grained control.")
    print()

    # Run each custom experiment
    embedding_result = run_custom_embedding_drift()
    alignment_result = run_custom_alignment_decay()
    loss_result = run_custom_loss_scaling()
    mem0_result = run_custom_mem0_retrieval()

    # Also demonstrate selective experiment running
    run_selected_experiments()

    # Final summary
    print("\n" + "=" * 60)
    print("Summary of All Custom Experiments")
    print("=" * 60)

    results = {
        "Embedding Drift": embedding_result,
        "Alignment Decay": alignment_result,
        "Loss Scaling": loss_result,
        "mem0 Retrieval": mem0_result,
    }

    consistent_count = sum(1 for r in results.values() if r.is_consistent)

    print(f"\nExperiments consistent with hypothesis: {consistent_count}/4")
    print("\nDetailed exponents:")
    for name, result in results.items():
        status = "CONSISTENT" if result.is_consistent else "INCONSISTENT"
        print(f"  {name}: {result.exponent:.4f} ({status})")


if __name__ == "__main__":
    main()
