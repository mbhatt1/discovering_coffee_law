# Query Drift Hypothesis Validation

Empirical validation framework for the Query Drift Hypothesis, which predicts systematic degradation patterns in transformer attention mechanisms over extended contexts.

## Overview

The Query Drift Hypothesis posits that in transformer attention mechanisms, query vectors undergo continuous drift as context accumulates while key vectors from earlier content remain static. This fundamental asymmetry leads to predictable degradation patterns that follow power law scaling.

### Core Predictions

1. **Alignment decay follows t^(-1/2)** - As query vectors drift from fixed historical keys, the alignment (cosine similarity) decays with the square root of time/position
2. **Loss scaling follows c^(-0.5)** - Model loss scales with context length following a power law with exponent approximately -0.5
3. **Memory retrieval degradation** - Long-term memory systems exhibit retrieval accuracy decay consistent with the t^(-1/2) prediction
4. **Embedding variance growth** - Query embedding variance grows as t^(0.5), characteristic of Brownian motion

### The Four Experiments

This framework implements four complementary experiments to validate the hypothesis:

| Experiment | Measures | Expected Exponent |
|------------|----------|-------------------|
| Embedding Drift | Query vector variance growth over token positions | ~0.5 (Hurst exponent H = 0.5) |
| Alignment Decay | Cosine similarity decay between query and historical keys | ~-0.5 |
| Loss Scaling | Perplexity/loss as function of context length | ~-0.5 |
| mem0 Retrieval | Memory retrieval accuracy vs. storage age | ~-0.5 |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/query_drift_validation.git
cd query_drift_validation

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Dependencies

- Python 3.10+
- numpy
- scipy
- openai
- matplotlib (for visualization)
- mem0ai (optional, for mem0 experiment)

### Basic Usage

```python
from query_drift import QueryDriftValidator, ExperimentConfig

# Create validator with default configuration
validator = QueryDriftValidator()

# Run all experiments
results = validator.run_all()

# Print summary
validator.print_summary(results)
```

### Quick Demo

```python
from query_drift import QueryDriftValidator, ExperimentConfig

# Use fast configuration for quick testing
config = ExperimentConfig.fast()
validator = QueryDriftValidator(config)

# Run experiments
results = validator.run_all()
```

## Experiment Descriptions

### 1. Embedding Drift Experiment

**Theoretical Background**: If query representations undergo Brownian-like drift, the variance of embeddings should grow linearly with position (or token count), yielding a Hurst exponent H = 0.5.

**Method**:
1. Start with a seed prompt
2. Generate multiple continuations using the language model
3. Extract embeddings at various token positions
4. Measure variance growth across continuations
5. Fit power law to variance vs. position data

**Key Metric**: Hurst exponent H (expected ~0.5)

### 2. Alignment Decay Experiment

**Theoretical Background**: As the query vector drifts from where it was when a key was encoded, the dot product (and thus attention weight) should decay predictably.

**Method**:
1. Encode an initial "key" context
2. Progressively inject distractor content
3. Measure cosine similarity between current query state and original key
4. Fit decay curve to similarity vs. distractor amount

**Key Metric**: Decay exponent (expected ~-0.5)

### 3. Loss Scaling Experiment

**Theoretical Background**: If alignment degradation follows t^(-0.5), the accumulated effect on loss should also exhibit power law scaling with context length.

**Method**:
1. Construct prompts of varying context lengths
2. Measure model loss/perplexity at each length
3. Fit power law to loss vs. context length

**Key Metric**: Loss scaling exponent (expected ~-0.5)

### 4. mem0 Retrieval Experiment

**Theoretical Background**: Long-term memory systems that rely on embedding similarity for retrieval should exhibit the same decay patterns as the underlying attention mechanism.

**Method**:
1. Store a series of memories with timestamps
2. Insert distractor memories
3. Query for original memories
4. Measure retrieval accuracy vs. memory age

**Key Metric**: Retrieval decay exponent (expected ~-0.5)

## Expected Output

When the hypothesis is validated, you should observe:

```
============================================================
Query Drift Hypothesis Validation Results
============================================================

Embedding Drift Experiment:
  Hurst exponent: 0.52 +/- 0.08
  R-squared: 0.94
  Status: CONSISTENT with hypothesis (H ~ 0.5)

Alignment Decay Experiment:
  Decay exponent: -0.48 +/- 0.06
  R-squared: 0.91
  Status: CONSISTENT with hypothesis (exponent ~ -0.5)

Loss Scaling Experiment:
  Scaling exponent: -0.51 +/- 0.05
  R-squared: 0.96
  Status: CONSISTENT with hypothesis (exponent ~ -0.5)

mem0 Retrieval Experiment:
  Decay exponent: -0.47 +/- 0.09
  R-squared: 0.88
  Status: CONSISTENT with hypothesis (exponent ~ -0.5)

============================================================
Overall: 4/4 experiments consistent with Query Drift Hypothesis
============================================================
```

The key indicator is that all exponents should be approximately **0.5** (or -0.5 for decay metrics), with R-squared values indicating good power law fits.

## Configuration Options

### Preset Configurations

```python
# Fast configuration (quick testing, ~5 minutes)
config = ExperimentConfig.fast()

# Default configuration (balanced, ~15 minutes)
config = ExperimentConfig()

# Thorough configuration (comprehensive, ~45 minutes)
config = ExperimentConfig.thorough()
```

### Custom Configuration

```python
from query_drift.config import (
    ExperimentConfig,
    EmbeddingDriftConfig,
    AlignmentDecayConfig,
    LossScalingConfig,
    Mem0RetrievalConfig,
    ModelConfig,
    OutputConfig,
)

config = ExperimentConfig(
    embedding_drift=EmbeddingDriftConfig(
        num_continuations=20,      # Number of parallel continuations
        max_tokens_drift=100,      # Maximum tokens to generate
        sample_positions=[10, 20, 30, 50, 70, 100],  # Positions to sample
    ),
    alignment_decay=AlignmentDecayConfig(
        num_distractor_rounds=3,   # Rounds of distractor injection
    ),
    loss_scaling=LossScalingConfig(
        context_lengths=[100, 200, 500, 1000, 2000, 4000],
        measurements_per_length=3,
    ),
    mem0_retrieval=Mem0RetrievalConfig(
        num_memories=20,
        distractor_multiplier=2,
        retrieval_limit=5,
    ),
    model=ModelConfig(
        embedding_model="text-embedding-3-small",
        completion_model="gpt-4o-mini",
    ),
    output=OutputConfig(
        output_dir="query_drift_results",
        save_plots=True,
        show_plots=True,
    ),
    # Select which experiments to run
    run_embedding_drift=True,
    run_alignment_decay=True,
    run_loss_scaling=True,
    run_mem0_retrieval=True,
)
```

### Configuration from JSON

```python
# Save configuration
config.to_json("my_config.json")

# Load configuration
config = ExperimentConfig.from_json("my_config.json")
```

## API Reference

### Core Classes

#### `QueryDriftValidator`

Main class for running validation experiments.

```python
class QueryDriftValidator:
    def __init__(self, config: ExperimentConfig = None):
        """Initialize validator with configuration."""

    def run_all(self) -> dict[str, ExperimentResult]:
        """Run all enabled experiments and return results."""

    def run_embedding_drift(self) -> ExperimentResult:
        """Run only the embedding drift experiment."""

    def run_alignment_decay(self) -> ExperimentResult:
        """Run only the alignment decay experiment."""

    def run_loss_scaling(self) -> ExperimentResult:
        """Run only the loss scaling experiment."""

    def run_mem0_retrieval(self) -> ExperimentResult:
        """Run only the mem0 retrieval experiment."""

    def print_summary(self, results: dict[str, ExperimentResult]) -> None:
        """Print formatted summary of results."""
```

#### `ExperimentConfig`

Configuration container for all experiment parameters.

```python
class ExperimentConfig:
    @classmethod
    def fast(cls) -> ExperimentConfig:
        """Create fast configuration for quick testing."""

    @classmethod
    def thorough(cls) -> ExperimentConfig:
        """Create thorough configuration for comprehensive validation."""

    @classmethod
    def from_json(cls, path: str) -> ExperimentConfig:
        """Load configuration from JSON file."""

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""

    def validate(self) -> list[str]:
        """Validate configuration and return warnings."""
```

#### `ExperimentResult`

Standardized result container for all experiments.

```python
@dataclass
class ExperimentResult:
    name: str                    # Experiment name
    exponent: float             # Fitted power law exponent
    exponent_std: float         # Standard error of exponent
    r_squared: float            # Goodness of fit
    is_consistent: bool         # Whether result supports hypothesis
    raw_data: dict              # Raw experimental data
    metadata: dict              # Additional metadata
```

### Experiment Classes

#### `EmbeddingDriftExperiment`

```python
class EmbeddingDriftExperiment:
    def __init__(self, config: ExperimentConfig):
        """Initialize with configuration."""

    def run(self, seed_prompt: str = None) -> ExperimentResult:
        """Run the embedding drift experiment."""
```

#### `AlignmentDecayExperiment`

```python
class AlignmentDecayExperiment:
    def __init__(self, config: ExperimentConfig):
        """Initialize with configuration."""

    def run(self, key_context: str = None) -> ExperimentResult:
        """Run the alignment decay experiment."""
```

#### `LossScalingExperiment`

```python
class LossScalingExperiment:
    def __init__(self, config: ExperimentConfig):
        """Initialize with configuration."""

    def run(self) -> ExperimentResult:
        """Run the loss scaling experiment."""
```

#### `Mem0RetrievalExperiment`

```python
class Mem0RetrievalExperiment:
    def __init__(self, config: ExperimentConfig):
        """Initialize with configuration."""

    def run(self) -> ExperimentResult:
        """Run the mem0 retrieval experiment."""
```

### Utility Functions

```python
from query_drift.utils import (
    fit_power_law,           # Fit power law to data
    cosine_similarity,       # Compute cosine similarity
    estimate_hurst_exponent, # Estimate Hurst exponent from variances
    EmbeddingClient,         # OpenAI embedding client with caching
)
```

## Interpreting Results

### Exponent Values

| Exponent Range | Interpretation |
|----------------|----------------|
| 0.45 - 0.55 | Strong support for hypothesis |
| 0.35 - 0.65 | Moderate support |
| < 0.35 or > 0.65 | Weak support or inconsistent |

### R-squared Values

| R-squared | Fit Quality |
|-----------|-------------|
| > 0.90 | Excellent fit |
| 0.80 - 0.90 | Good fit |
| 0.70 - 0.80 | Acceptable fit |
| < 0.70 | Poor fit (results may be unreliable) |

### Common Issues

1. **Low R-squared**: May indicate insufficient data points or high noise. Try increasing `num_continuations` or `measurements_per_length`.

2. **Exponent far from 0.5**: Could indicate:
   - Insufficient context length range
   - Model-specific behavior
   - Implementation artifacts

3. **High variance in exponent**: Increase sample sizes in configuration.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{query_drift_validation,
  title = {Query Drift Hypothesis Validation Framework},
  year = {2024},
  url = {https://github.com/your-org/query_drift_validation}
}
```
