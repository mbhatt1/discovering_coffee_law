"""
Configuration management for Query Drift experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import os
import json


@dataclass
class EmbeddingDriftConfig:
    """Configuration for the embedding drift experiment."""
    num_continuations: int = 20
    max_tokens_drift: int = 100
    sample_positions: list[int] = field(default_factory=lambda: [10, 20, 30, 50, 70, 100])
    temperature: float = 1.0


@dataclass
class AlignmentDecayConfig:
    """Configuration for the alignment decay experiment."""
    num_distractor_rounds: int = 3
    distractors: list[str] = field(default_factory=lambda: [
        "The weather today is quite pleasant. ",
        "Meanwhile, in other news, technology continues to advance. ",
        "Speaking of which, the history of computing is fascinating. ",
        "One might also consider the philosophical implications. ",
        "There are many factors to consider in such discussions. ",
        "Furthermore, recent developments have shown interesting trends. ",
        "It's worth noting that various perspectives exist on this topic. ",
        "Additionally, the economic impact cannot be ignored. ",
        "From a historical standpoint, these events are significant. ",
        "The cultural implications are equally important to consider. ",
    ])


@dataclass
class LossScalingConfig:
    """Configuration for the loss scaling experiment."""
    context_lengths: list[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000, 4000])
    measurements_per_length: int = 3
    continuation_prompt: str = " Therefore,"


@dataclass
class Mem0RetrievalConfig:
    """Configuration for the mem0 retrieval experiment."""
    num_memories: int = 20
    distractor_multiplier: int = 2
    retrieval_limit: int = 5
    memory_delay_seconds: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    embedding_model: str = "text-embedding-3-small"
    completion_model: str = "gpt-4o-mini"
    api_timeout: int = 60


@dataclass
class OutputConfig:
    """Configuration for output and visualization."""
    output_dir: str = "query_drift_results"
    save_plots: bool = True
    show_plots: bool = True
    plot_dpi: int = 150
    plot_format: str = "png"
    save_json: bool = True
    verbose: bool = True


@dataclass
class ExperimentConfig:
    """
    Master configuration for all Query Drift experiments.

    Provides sensible defaults that can be overridden for specific use cases.
    """
    # Sub-configurations
    embedding_drift: EmbeddingDriftConfig = field(default_factory=EmbeddingDriftConfig)
    alignment_decay: AlignmentDecayConfig = field(default_factory=AlignmentDecayConfig)
    loss_scaling: LossScalingConfig = field(default_factory=LossScalingConfig)
    mem0_retrieval: Mem0RetrievalConfig = field(default_factory=Mem0RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Experiment selection
    run_embedding_drift: bool = True
    run_alignment_decay: bool = True
    run_loss_scaling: bool = True
    run_mem0_retrieval: bool = True

    @classmethod
    def fast(cls) -> "ExperimentConfig":
        """Create a fast configuration for quick testing."""
        return cls(
            embedding_drift=EmbeddingDriftConfig(
                num_continuations=5,
                max_tokens_drift=50,
                sample_positions=[10, 25, 50]
            ),
            alignment_decay=AlignmentDecayConfig(num_distractor_rounds=2),
            loss_scaling=LossScalingConfig(
                context_lengths=[100, 500, 1000],
                measurements_per_length=2
            ),
            mem0_retrieval=Mem0RetrievalConfig(num_memories=10, distractor_multiplier=1),
            output=OutputConfig(verbose=True)
        )

    @classmethod
    def thorough(cls) -> "ExperimentConfig":
        """Create a thorough configuration for comprehensive validation."""
        return cls(
            embedding_drift=EmbeddingDriftConfig(
                num_continuations=30,
                max_tokens_drift=150,
                sample_positions=[10, 20, 30, 50, 70, 100, 130, 150]
            ),
            alignment_decay=AlignmentDecayConfig(num_distractor_rounds=5),
            loss_scaling=LossScalingConfig(
                context_lengths=[50, 100, 200, 500, 1000, 2000, 4000, 8000],
                measurements_per_length=5
            ),
            mem0_retrieval=Mem0RetrievalConfig(num_memories=30, distractor_multiplier=3),
            output=OutputConfig(verbose=True)
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create configuration from a dictionary."""
        config = cls()

        if "embedding_drift" in data:
            config.embedding_drift = EmbeddingDriftConfig(**data["embedding_drift"])
        if "alignment_decay" in data:
            config.alignment_decay = AlignmentDecayConfig(**data["alignment_decay"])
        if "loss_scaling" in data:
            config.loss_scaling = LossScalingConfig(**data["loss_scaling"])
        if "mem0_retrieval" in data:
            config.mem0_retrieval = Mem0RetrievalConfig(**data["mem0_retrieval"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])

        for key in ["run_embedding_drift", "run_alignment_decay",
                    "run_loss_scaling", "run_mem0_retrieval"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return {
            "embedding_drift": asdict(self.embedding_drift),
            "alignment_decay": asdict(self.alignment_decay),
            "loss_scaling": asdict(self.loss_scaling),
            "mem0_retrieval": asdict(self.mem0_retrieval),
            "model": asdict(self.model),
            "output": asdict(self.output),
            "run_embedding_drift": self.run_embedding_drift,
            "run_alignment_decay": self.run_alignment_decay,
            "run_loss_scaling": self.run_loss_scaling,
            "run_mem0_retrieval": self.run_mem0_retrieval,
        }

    def to_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if not os.environ.get("OPENAI_API_KEY"):
            warnings.append("OPENAI_API_KEY environment variable not set")

        if self.embedding_drift.num_continuations < 3:
            warnings.append("num_continuations < 3 may not provide reliable variance estimates")

        if len(self.loss_scaling.context_lengths) < 3:
            warnings.append("Fewer than 3 context lengths may not provide reliable power law fits")

        return warnings
