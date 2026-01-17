#!/usr/bin/env python3
"""
Simple validation script demonstrating the Query Drift Hypothesis validation framework.

This script runs all four experiments with a fast configuration suitable for demos.
"""

from query_drift import QueryDriftValidator, ExperimentConfig


def main():
    """Run the Query Drift Hypothesis validation."""
    print("Query Drift Hypothesis Validation")
    print("=" * 50)
    print()

    # Create fast configuration for demo purposes
    # For thorough validation, use ExperimentConfig.thorough() or default ExperimentConfig()
    config = ExperimentConfig.fast()

    # Validate configuration and print any warnings
    warnings = config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    # Create validator
    validator = QueryDriftValidator(config)

    # Run all experiments
    print("Running all experiments...")
    print()
    results = validator.run_all()

    # Print summary
    print()
    validator.print_summary(results)

    # Return results for further analysis if needed
    return results


if __name__ == "__main__":
    results = main()
