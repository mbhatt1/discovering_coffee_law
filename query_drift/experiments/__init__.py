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
"""

from .base import BaseExperiment, ExperimentResult
from .embedding_drift import EmbeddingDriftExperiment
from .alignment_decay import AlignmentDecayExperiment
from .loss_scaling import LossScalingExperiment
from .mem0_retrieval import Mem0RetrievalExperiment

__all__ = [
    "ExperimentResult",
    "BaseExperiment",
    "EmbeddingDriftExperiment",
    "AlignmentDecayExperiment",
    "LossScalingExperiment",
    "Mem0RetrievalExperiment",
]
