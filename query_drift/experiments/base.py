"""
Base classes for Query Drift experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional
import json
import time
import traceback

from openai import OpenAI

from ..config import ExperimentConfig
from ..utils.embeddings import EmbeddingClient
from ..utils.logging import ExperimentLogger


@dataclass
class ExperimentResult:
    """
    Container for experiment results.

    Stores all relevant data from an experiment run including
    metrics, predictions, errors, and timing information.
    """

    experiment_name: str = ""
    name: str = ""  # Alias for experiment_name
    success: bool = False
    data: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def __post_init__(self):
        # Sync name and experiment_name
        if self.experiment_name and not self.name:
            self.name = self.experiment_name
        elif self.name and not self.experiment_name:
            self.experiment_name = self.name

    def to_dict(self) -> dict:
        """Convert result to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert result to a JSON string."""
        def convert(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        return json.dumps(self.to_dict(), indent=indent, default=convert)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Create an ExperimentResult from a dictionary."""
        return cls(
            experiment_name=data.get("experiment_name", data.get("name", "unknown")),
            name=data.get("name", data.get("experiment_name", "unknown")),
            success=data.get("success", False),
            data=data.get("data", {}),
            metrics=data.get("metrics", {}),
            predictions=data.get("predictions", {}),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            error=data.get("error", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    def summary(self) -> str:
        """Generate a human-readable summary of the result."""
        lines = [
            f"Experiment: {self.experiment_name or self.name}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Duration: {self.duration_seconds:.2f}s",
        ]

        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)


class BaseExperiment(ABC):
    """
    Abstract base class for all Query Drift experiments.

    Provides common functionality for running experiments including
    API interactions, logging, error handling, and result management.
    """

    experiment_name: str = "base_experiment"

    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[OpenAI] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        logger: Optional[ExperimentLogger] = None,
    ):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration
            client: OpenAI client (creates new one if not provided)
            embedding_client: Embedding client (creates new one if not provided)
            logger: Optional experiment logger
        """
        self.config = config
        self.client = client or OpenAI()
        self.embedding_client = embedding_client or EmbeddingClient(
            client=self.client,
            model=config.model.embedding_model
        )
        self.logger = logger
        self.result: Optional[ExperimentResult] = None
        self._start_time: Optional[float] = None

    @property
    def name(self) -> str:
        """Return the name of the experiment."""
        return self.experiment_name

    @abstractmethod
    def run(self) -> ExperimentResult:
        """
        Run the experiment.

        Returns:
            ExperimentResult containing all experiment data and metrics.
        """
        pass

    @abstractmethod
    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots for the experiment results.

        Args:
            save_path: Optional path to save the plot.
        """
        pass

    def log(self, message: str) -> None:
        """Log a message."""
        if self.config.output.verbose:
            print(f"[{self.experiment_name}] {message}")
        if self.logger:
            self.logger.info(message)

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a text using the configured embedding model.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        result = self.embedding_client.get_embedding(text)
        return list(result.embedding)

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a completion using the configured completion model.

        Args:
            prompt: The prompt to complete.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            logprobs: Whether to return log probabilities.

        Returns:
            Dictionary containing the completion response data.
        """
        kwargs = {
            "model": self.config.model.completion_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if logprobs:
            kwargs["logprobs"] = True

        response = self.client.chat.completions.create(**kwargs)

        result = {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
        }

        if logprobs and response.choices[0].logprobs:
            result["logprobs"] = response.choices[0].logprobs

        return result
