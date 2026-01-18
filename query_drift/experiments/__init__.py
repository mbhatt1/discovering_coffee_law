"""
Query Drift Experiments

This module contains experiment implementations for validating
the Query Drift Hypothesis predictions.

Exports:
    ExperimentResult: Standardized result container for all experiments
    BaseExperiment: Abstract base class for all experiments
    EmbeddingDriftExperiment: Measures embedding variance growth over token positions
    AlignmentDecayExperiment: Measures alignment decay between queries and historical keys
    LossScalingExperiment: Measures how model loss scales with context length
    Mem0RetrievalExperiment: Measures memory retrieval decay in mem0
    StressTestRetrievalExperiment: Extreme distractor counts to break ceiling effect
    EntropyDriftExperiment: Controls for LayerNorm artifact via entropy measurement
    ExtendedContextExperiment: Tests variance saturation at 10k+ token contexts
    DenseSamplingExperiment: Dense sampling + prompt diversity (hierarchical) for COFFEE Law
    LostInMiddleExperiment: Liu et al. (2023) protocol for position-dependent retrieval
    InternalStatesExperiment: Open-weight internal-state replication (Llama/Mistral)
"""

from .base import BaseExperiment, ExperimentResult
from .embedding_drift import EmbeddingDriftExperiment
from .alignment_decay import AlignmentDecayExperiment
from .loss_scaling import LossScalingExperiment
from .mem0_retrieval import Mem0RetrievalExperiment
from .stress_test_retrieval import StressTestRetrievalExperiment
from .entropy_drift import EntropyDriftExperiment
from .extended_context import ExtendedContextExperiment
from .dense_sampling import DenseSamplingExperiment
from .lost_in_middle import LostInMiddleExperiment
from .internal_states import InternalStatesExperiment

__all__ = [
    "ExperimentResult",
    "BaseExperiment",
    "EmbeddingDriftExperiment",
    "AlignmentDecayExperiment",
    "LossScalingExperiment",
    "Mem0RetrievalExperiment",
    "StressTestRetrievalExperiment",
    "EntropyDriftExperiment",
    "ExtendedContextExperiment",
    "DenseSamplingExperiment",
    "LostInMiddleExperiment",
    "InternalStatesExperiment",
]
