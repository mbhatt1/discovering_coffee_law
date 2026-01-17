"""
Query Drift Hypothesis Validation Package

Empirical validation of the Query Drift Hypothesis which predicts:
1. Alignment decay follows t^(-1/2) as query vectors drift from fixed historical keys
2. Loss scaling follows c^(-0.5) with context length
3. Memory retrieval degradation through mem0's long-term memory system
"""

__version__ = "0.1.0"
__author__ = "Query Drift Research"

from .config import ExperimentConfig
from .validator import QueryDriftValidator

__all__ = ["ExperimentConfig", "QueryDriftValidator", "__version__"]
