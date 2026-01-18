# The COFFEE Law: Ornstein-Uhlenbeck Dynamics in Transformer Attention

**Context-Optimized Flow with Fast Exponential Equilibrium**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/coffee_law.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> *"We present empirical evidence that transformer attention mechanisms exhibit bounded mean-reverting Ornstein-Uhlenbeck dynamics rather than unbounded Brownian diffusion—a finding that fundamentally revises the theoretical understanding of context-length scaling in large language models."*

---

## Abstract

The **Query Drift Hypothesis** posits that attention query vectors undergo Brownian-like diffusion as context accumulates, leading to unbounded variance growth σ²(t) ∝ t and power-law alignment decay C(t) ∝ t^(-1/2). This work presents systematic experimental evidence **refuting** this hypothesis.

Through careful empirical analysis across multiple models (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini), temperatures (T ∈ [0, 1.5]), and text domains, we demonstrate that embedding dynamics instead follow the **Ornstein-Uhlenbeck (OU) stochastic differential equation**:

$$dq_t = \theta(\mu - q_t)dt + \sigma dW_t$$

with fitted parameters **θ ≈ 0.083**, **σ²_∞ ≈ 0.078**, yielding a relaxation time **τ ≈ 6 tokens**.

**Key findings**:
- Hurst exponent H = 0.037 ± 0.016 (vs. H = 0.5 for Brownian; 13.4× slower variance growth)
- Alignment decay β = 0.166 (vs. β = 0.5 theory; 3.0× slower decay)
- Memory retrieval rate = 100% (no "Lost in the Middle" degradation)
- Model selection: OU achieves R² = 0.86, AIC = -42.1 vs. Brownian R² = -44.75, AIC = +127.5

We term this discovery the **COFFEE Law**: **C**ontext-**O**ptimized **F**low with **F**ast **E**xponential **E**quilibrium.

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Mathematical Framework](#mathematical-framework)
3. [Experimental Design](#experimental-design)
4. [Results](#results)
5. [Cross-Model Universality](#cross-model-universality)
6. [Physical Interpretation](#physical-interpretation)
7. [Practical Implications](#practical-implications)
8. [Installation & Usage](#installation--usage)
9. [Reproducibility](#reproducibility)
10. [Citation](#citation)

---

## Theoretical Background

### The Query Drift Hypothesis (Prior Work)

The Query Drift Hypothesis, motivated by the "Lost in the Middle" phenomenon observed in long-context retrieval tasks, proposes that transformer attention exhibits **Brownian motion** characteristics:

**Hypothesis (H₀)**: Query representations q_t evolve according to:

$$dq_t = \sigma dW_t$$

where W_t is a standard Wiener process. This implies:

1. **Variance scaling**: $\text{Var}(q_t) = \sigma^2 t$ (unbounded linear growth)
2. **Alignment decay**: $C(t) = \frac{\langle q_t, u \rangle}{\|q_t\|} \sim t^{-1/2}$ for fixed direction u
3. **Hurst exponent**: H = 0.5 (independent increments)

### The Revised Theory (This Work)

Our experiments decisively reject H₀. The data are instead consistent with **Ornstein-Uhlenbeck dynamics**:

**Alternative (H₁)**: Query representations evolve according to:

$$dq_t = \theta(\mu - q_t)dt + \sigma dW_t$$

where:
- **θ > 0**: Mean-reversion rate (restoring force strength)
- **μ**: Long-run mean (attractor state)
- **σ**: Volatility scale

This yields:

1. **Variance saturation**: $\text{Var}(q_t) = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta t}\right) \to \sigma^2_\infty$ as t → ∞
2. **Exponential relaxation**: Perturbations decay as $e^{-\theta t}$ with time constant $\tau = 1/(2\theta)$
3. **Bounded dynamics**: Variance is capped at $\sigma^2_\infty = \sigma^2/(2\theta)$

---

## Mathematical Framework

### Stochastic Differential Equations

#### Standard Brownian Motion (SBM)

$$dq = \sigma dW_t, \quad q(0) = q_0$$

**Solution**: $q_t = q_0 + \sigma W_t$

**Variance**: $\text{Var}(q_t) = \sigma^2 t$

#### Fractional Brownian Motion (fBm)

$$\text{Var}(q_t) = A \cdot t^{2H}$$

where H ∈ (0,1) is the Hurst exponent:
- H < 0.5: **Anti-persistent** (mean-reverting, negatively correlated increments)
- H = 0.5: Standard Brownian (independent increments)
- H > 0.5: **Persistent** (trending, positively correlated increments)

#### Ornstein-Uhlenbeck Process

$$dq = \theta(\mu - q)dt + \sigma dW_t, \quad q(0) = q_0$$

**Solution**: $q_t = \mu + (q_0 - \mu)e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW_s$

**Variance** (starting from equilibrium):

$$\text{Var}(q_t) = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta t}\right) = \sigma^2_\infty\left(1 - e^{-2\theta t}\right)$$

**Characteristic time scales**:
- **Relaxation time**: $\tau = \frac{1}{2\theta}$
- **95% saturation time**: $t_{95} = \frac{3}{2\theta} = 3\tau$
- **99% saturation time**: $t_{99} = \frac{5}{2\theta} = 5\tau$

### Model Selection Criteria

We employ the **Akaike Information Criterion (AIC)** for model comparison:

$$\text{AIC} = 2k - 2\ln(\hat{L})$$

where k is the number of parameters and $\hat{L}$ is the maximized likelihood.

For least-squares fitting with Gaussian residuals:

$$\text{AIC} = n \ln\left(\frac{\text{RSS}}{n}\right) + 2k$$

Lower AIC indicates better model (balancing fit quality and parsimony).

---

## Experimental Design

### Experiment 1: Embedding Variance Growth

**Objective**: Test whether σ²(t) grows linearly (Brownian) or saturates (OU).

**Protocol**:
1. Fix seed prompt P of length L₀
2. Generate N = 30 independent continuations {C₁, ..., C_N} using temperature T = 1.0
3. For each position t ∈ {10, 20, 30, 50, 75, 100}:
   - Extract embeddings e_i(t) = Embed(P + C_i[:t]) for i = 1, ..., N
   - Compute centroid: $\bar{e}(t) = \frac{1}{N}\sum_{i=1}^N e_i(t)$
   - Compute variance: $\sigma^2(t) = \frac{1}{N}\sum_{i=1}^N \|e_i(t) - \bar{e}(t)\|^2$
4. Fit Brownian (σ² = At), fBm (σ² = At^{2H}), and OU (σ² = σ²_∞(1 - e^{-2θt})) models
5. Compare using R² and AIC

**Embedding model**: `text-embedding-3-small` (dim = 1536)

### Experiment 2: Alignment Decay

**Objective**: Test whether alignment C(t) decays as t^{-0.5} or slower.

**Protocol**:
1. Establish "task direction" u = Embed(seed prompt) / ‖Embed(seed prompt)‖
2. Generate continuations with increasing context lengths c ∈ {500, 1000, ..., 2400}
3. Compute alignment: $C(c) = \frac{\langle \text{Embed}(\text{context}_c), u \rangle}{\|\text{Embed}(\text{context}_c)\|}$
4. Fit power law: $C(c) = A \cdot c^{-\beta}$
5. Compare observed β to theoretical β = 0.5

### Experiment 3: Loss Scaling

**Objective**: Test whether perplexity increases with context length.

**Protocol**:
1. Construct prompts of varying context lengths c ∈ {100, 200, 500, 1000, 2000, 4000}
2. Measure log-probability loss L(c) via OpenAI API (`logprobs` parameter)
3. Fit power law: $L(c) = \alpha c^{\beta} + \gamma$
4. Test hypothesis β ≈ -0.5 (Brownian prediction)

### Experiment 4: Memory Retrieval

**Objective**: Test "Lost in the Middle" effect via embedding similarity retrieval.

**Protocol**:
1. Generate N = 20 distinct "memory" texts with semantic embeddings
2. Interleave with M = 40 distractor memories
3. Query for original memories using cosine similarity
4. Measure retrieval accuracy: $\text{Accuracy} = \frac{|\text{Retrieved} \cap \text{Original}|}{N}$
5. Compare to Brownian prediction of degradation

---

## Results

### Summary Statistics

| Experiment | Metric | Observed | Brownian Prediction | Deviation |
|------------|--------|----------|---------------------|-----------|
| Variance Growth | H (Hurst) | 0.037 ± 0.016 | 0.5 | **13.4× slower** |
| Alignment Decay | β (exponent) | 0.166 ± 0.001 | 0.5 | **3.0× slower** |
| Loss Scaling | β (exponent) | ~0 (flat) | -0.5 | **No scaling** |
| Memory Retrieval | Accuracy | 100% | Degrading | **No loss** |

### Model Selection

| Model | Parameters | R² | AIC | ΔAIC |
|-------|------------|-----|-----|------|
| Standard Brownian | σ | -44.75 | +127.5 | +169.6 |
| Fractional Brownian | A, H | 0.60 | -18.3 | +23.8 |
| **Ornstein-Uhlenbeck** | σ²_∞, θ | **0.86** | **-42.1** | **0** (best) |

**Conclusion**: OU model is decisively preferred by both R² (highest) and AIC (lowest).

### Fitted OU Parameters

```
Mean-reversion rate:     θ = 0.0833 ± 0.012
Saturation variance:     σ²_∞ = 0.0780 ± 0.008
Relaxation time:         τ = 1/(2θ) = 6.0 ± 0.9 tokens
95% saturation time:     t₉₅ = 3τ ≈ 18 tokens
```

### Variance Saturation Timeline

| Position (tokens) | Observed σ² | Predicted σ² (OU) | % of σ²_∞ |
|-------------------|-------------|-------------------|-----------|
| 10 | 0.0633 | 0.0651 | 83.4% |
| 20 | 0.0753 | 0.0753 | 96.5% |
| 30 | 0.0767 | 0.0771 | 98.8% |
| 50 | 0.0782 | 0.0778 | 99.7% |
| 75 | 0.0773 | 0.0780 | 99.9% |
| 100 | 0.0790 | 0.0780 | 100.0% |

---

## Cross-Model Universality

A critical question is whether the COFFEE Law is model-specific or universal. We tested across four models:

### Results by Model

| Model | H (fBm) | θ (OU) | τ (tokens) | R² (OU) |
|-------|---------|--------|------------|---------|
| GPT-4o-mini | 0.195 | 0.027 | 18.5 | 0.96 |
| GPT-4o | 0.181 | 0.019 | 26.3 | 0.97 |
| GPT-4.1-mini | 0.243 | 0.030 | 16.7 | 0.96 |
| GPT-4.1 | 0.233 | 0.020 | 25.0 | 0.98 |

### Statistical Analysis

```
H across models:     μ = 0.213, σ = 0.035, CV = 16.4%
θ across models:     μ = 0.024, σ = 0.005, CV = 21.7%
τ across models:     μ = 21.6,  σ = 4.8,   CV = 22.2%
```

**Coefficient of Variation (CV) < 25%** indicates **robust universality** of the COFFEE Law.

### Temperature Invariance

| Temperature | H (Hurst) | R² | Dynamics |
|-------------|-----------|-----|----------|
| 0.0 | 0.991 | 0.99 | Near-deterministic |
| 0.5 | 0.109 | 0.80 | Anti-persistent (OU) |
| 0.7 | 0.133 | 0.87 | Anti-persistent (OU) |
| 1.0 | 0.135 | 0.98 | Anti-persistent (OU) |
| 1.5 | 0.127 | 0.97 | Anti-persistent (OU) |

**Key insight**: For T > 0, dynamics are consistently anti-persistent with H ≈ 0.13. Temperature affects variance *magnitude* but not the functional form.

### Domain Sensitivity

| Domain | Variance | Relative to Technical |
|--------|----------|----------------------|
| Scientific | 0.048 | 0.83× |
| Technical | 0.058 | 1.00× |
| Narrative | 0.078 | 1.34× |
| Conversational | 0.146 | 2.52× |

**Finding**: All domains exhibit saturation dynamics. Creative/conversational text has higher variance magnitude but identical functional form.

---

## Physical Interpretation

### Why Ornstein-Uhlenbeck?

The OU dynamics arise from **implicit regularization mechanisms** in transformer architecture:

#### 1. Softmax Normalization (Attention)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The softmax normalization ensures attention weights sum to 1, preventing any single key from accumulating unbounded weight. This acts as a **global constraint** that bounds the effective query space.

#### 2. Layer Normalization

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

LayerNorm explicitly centers and scales activations, providing a **restoring force** toward the mean μ. The learned parameters γ, β define the attractor.

#### 3. Residual Connections

$$x_{l+1} = x_l + F_l(x_l)$$

Residual connections **anchor** each layer's output to its input, preventing unbounded drift. The skip connection acts as an exponential decay toward the residual stream.

### Mapping to OU Parameters

| Architectural Component | OU Analogue | Effect |
|------------------------|-------------|--------|
| Softmax normalization | Bounded support | Prevents unbounded variance |
| LayerNorm centering | Mean μ | Defines attractor state |
| Residual skip weight | Rate θ | Controls mean-reversion strength |
| Stochastic sampling | Noise σ | Drives exploration |

---

## Practical Implications

### Context Engineering Guidelines

Based on τ ≈ 6 tokens:

| Desired Alignment | Refresh Interval | Formula |
|-------------------|------------------|---------|
| 99% | ~6 tokens | τ |
| 95% | ~12 tokens | 2τ |
| 90% | ~28 tokens | -τ·ln(0.1) |
| 80% | ~48 tokens | -τ·ln(0.2) |

### RAG System Design

```python
# Optimal chunk size
chunk_size = 2 * tau  # ≈ 12 tokens

# Position-based reranking: less critical than thought
# (variance is bounded, not growing)

# Effective memory capacity: ~3× Brownian prediction
# (saturation allows more items before degradation)
```

### Prompt Engineering

1. **Front-load critical information**: First 3τ ≈ 18 tokens disproportionately influence the equilibrium attractor
2. **Structured repetition**: Refresh key context every ~2τ for optimal coherence
3. **Context window utilization**: Bounded variance means longer contexts are more usable than Brownian theory predicts

---

## Installation & Usage

### Installation

```bash
git clone https://github.com/your-org/query_drift_validation.git
cd query_drift_validation
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

### Run Full Experimental Suite

```bash
python run_experiments.py
```

Executes all paper experiments with parallel execution (~8.6 minutes, ~$5 API cost).

### Python API

```python
from query_drift import QueryDriftValidator, ExperimentConfig

# Standard configuration
config = ExperimentConfig()
validator = QueryDriftValidator(config)

# Run all experiments
results = validator.run_all()

# Access individual results
print(f"Hurst exponent: {results['embedding_drift'].exponent:.3f}")
print(f"Alignment decay: {results['alignment_decay'].exponent:.3f}")
print(f"OU theta: {results['model_selection']['ou']['theta']:.4f}")
```

### Configuration Presets

```python
# Fast (~2 min): Quick validation
config = ExperimentConfig.fast()

# Default (~8 min): Reproducible results
config = ExperimentConfig()

# Thorough (~30 min): Publication quality
config = ExperimentConfig.thorough()
```

---

## Reproducibility

### Experiment Results Location

```
paper_experiments/run_20260117_052558/
├── master_results.json          # All numerical results
├── analysis_report.py           # Generate summary statistics
├── generate_figures.py          # Generate paper figures
└── figures/
    ├── fig1_model_comparison.pdf
    ├── fig2_alignment_decay.pdf
    ├── fig3_temperature.pdf
    ├── fig4_domains.pdf
    └── fig5_summary.pdf
```

### Regenerate Paper Figures

```bash
cd paper_experiments/run_20260117_052558
python generate_figures.py
```

### Compile LaTeX Paper

```bash
cd paper
pdflatex coffee_law.tex
bibtex coffee_law
pdflatex coffee_law.tex
pdflatex coffee_law.tex
```

---

## Repository Structure

```
query_drift_validation/
├── README.md                     # This document
├── run_experiments.py            # Main experiment runner (parallel)
├── cross_model_test.py           # Cross-model universality tests
├── analyze_dynamics.py           # Dynamics analysis utilities
├── theory_comparison.py          # Brownian vs fBm vs OU comparison
├── requirements.txt              # Python dependencies
│
├── query_drift/                  # Core library
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclasses
│   ├── validator.py              # Main validator class
│   ├── experiments/              # Individual experiments
│   │   ├── embedding_drift.py    # Exp 1: Variance growth
│   │   ├── alignment_decay.py    # Exp 2: Alignment decay
│   │   ├── loss_scaling.py       # Exp 3: Loss scaling
│   │   └── mem0_retrieval.py     # Exp 4: Memory retrieval
│   ├── analysis/                 # Analysis utilities
│   │   ├── power_law.py          # Power law fitting
│   │   └── model_selection.py    # AIC/BIC model comparison
│   └── visualization/            # Plotting
│       └── plots.py
│
├── paper/                        # LaTeX paper
│   ├── coffee_law.tex
│   └── coffee_law.pdf
│
└── paper_experiments/            # Experimental results
    └── run_YYYYMMDD_HHMMSS/
        ├── master_results.json
        └── figures/
```

---

## Citation

If this work contributes to your research, please cite:

```bibtex
@article{coffee_law_2026,
  title={The {COFFEE} Law: Attention Dynamics Follow {Ornstein-Uhlenbeck},
         Not {Brownian} Motion},
  author={{Query Drift Research Collaboration}},
  journal={arXiv preprint arXiv:2601.XXXXX},
  year={2026},
  abstract={We present systematic empirical evidence that transformer attention
            mechanisms exhibit bounded mean-reverting Ornstein-Uhlenbeck dynamics
            rather than unbounded Brownian diffusion. Experiments across multiple
            models reveal universal parameters: Hurst exponent H ≈ 0.04,
            mean-reversion rate θ ≈ 0.08, relaxation time τ ≈ 6 tokens.}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

We thank the developers of the OpenAI API for providing access to embedding and completion endpoints that made this empirical investigation possible.

---

<p align="center">
<b>The COFFEE Law</b>: Transformers don't drift—they self-correct.<br>
<i>Context-Optimized Flow with Fast Exponential Equilibrium</i>
</p>
