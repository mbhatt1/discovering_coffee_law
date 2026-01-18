# Quick Start: Bulletproof COFFEE Law Validation

## TL;DR

```bash
cd query_drift_validation
export OPENAI_API_KEY='your-key-here'
python run_bulletproof_experiments.py
```

Expected duration: 1-2 hours | Estimated cost: $5-15

## What This Does

Runs 4 critical experiments that address major paper vulnerabilities:

1. **Stress Test Retrieval** - Breaks 100% ceiling with 1000 distractors
2. **Entropy Drift** - Controls for LayerNorm artifact  
3. **Extended Context** - Tests saturation at 16k tokens
4. **Enhanced Embedding Drift** - 5 trials for statistical rigor

## Prerequisites

1. Python 3.8+ with dependencies:
```bash
pip install -r requirements.txt
```

2. OpenAI API key with sufficient credits ($5-15 recommended)

3. Approximately 2 hours of runtime

## Step-by-Step

### 1. Set API Key

```bash
export OPENAI_API_KEY='sk-...'
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

### 2. Navigate to Directory

```bash
cd /path/to/query_drift_validation
```

### 3. Run Experiments

```bash
python run_bulletproof_experiments.py
```

You'll see output like:
```
================================================================================
  BULLETPROOF COFFEE LAW VALIDATION SUITE
================================================================================

Addressing Critical Vulnerabilities:
  1. ✓ Stress Test Retrieval (200-1000 distractors)
  2. ✓ Entropy Drift (LayerNorm control)
  3. ✓ Extended Context (10k+ tokens)
  4. ✓ Increased Trials (5 instead of 2)
  5. ✓ OU Process Validation (exponential fits)
--------------------------------------------------------------------------------

Output directory: bulletproof_results_20260117_120000

================================================================================
EXPERIMENT 1: STRESS TEST RETRIEVAL
================================================================================
Goal: Break the 100% ceiling and measure OU decay curve
...
```

### 4. Monitor Progress

The script will:
- Create a timestamped output directory
- Run each experiment sequentially
- Save results and plots after each experiment
- Generate a comprehensive summary report at the end

**Progress Indicators:**
- Stress Test: ~15-30 minutes (tests 5 distractor levels × 5 trials)
- Entropy Drift: ~10-15 minutes (generates 30 sequences with logprobs)
- Extended Context: ~30-60 minutes (most expensive - generates long contexts)
- Enhanced Embedding: ~15-20 minutes (5 trials × 30 continuations)

### 5. Review Results

After completion, check:
```bash
cd bulletproof_results_TIMESTAMP/
cat SUMMARY_REPORT.md
```

The report shows:
- Whether each vulnerability was addressed
- Key metrics (R², decay rates, Hurst exponents)
- Statistical summaries
- Validation status

### 6. Examine Detailed Results

```bash
# View individual experiment results
cat stress_test/results.json
cat entropy_drift/results.json
cat extended_context/results.json
cat embedding_drift_aggregated.json

# View plots
open stress_test/plot.png
open entropy_drift/plot.png
open extended_context/plot.png
open embedding_drift_trial_1/plot.png
```

## Output Structure

```
bulletproof_results_TIMESTAMP/
├── SUMMARY_REPORT.md              # Human-readable summary
├── SUMMARY_REPORT.txt             # Plain text version
│
├── stress_test/
│   ├── results.json               # Numerical results
│   └── plot.png                   # Degradation curve
│
├── entropy_drift/
│   ├── results.json               # Entropy measurements
│   └── plot.png                   # Entropy vs position
│
├── extended_context/
│   ├── results.json               # Variance at long contexts
│   └── plot.png                   # Saturation curve
│
├── embedding_drift_trial_1/       # Trial 1
│   ├── results.json
│   └── plot.png
├── embedding_drift_trial_2/       # Trial 2
├── embedding_drift_trial_3/       # Trial 3
├── embedding_drift_trial_4/       # Trial 4
├── embedding_drift_trial_5/       # Trial 5
│
└── embedding_drift_aggregated.json # Aggregated statistics
```

## Interrupting and Resuming

**To interrupt:** Press `Ctrl+C`

The script will save completed experiments and generate a partial summary.

**To resume:** Unfortunately, experiments must be re-run from scratch. Consider:
1. Running experiments individually (see Custom Usage below)
2. Commenting out completed experiments in `run_bulletproof_experiments.py`

## Cost Optimization

To reduce costs during testing:

### Option 1: Quick Test Run

Modify `run_bulletproof_experiments.py` to use fewer samples:

```python
# In run_stress_test()
distractor_counts = [50, 100, 200]  # Instead of [50, 100, 200, 500, 1000]
num_trials = 2                       # Instead of 5

# In run_extended_context()  
context_lengths = [1000, 2000, 4000, 8000]  # Instead of up to 16000
num_trials = 2                               # Instead of 3

# In run_enhanced_embedding_drift()
range(3)  # Instead of range(5)
```

### Option 2: Run Individual Experiments

```python
from query_drift.experiments import StressTestRetrievalExperiment
from query_drift.config import ExperimentConfig

config = ExperimentConfig(...)
experiment = StressTestRetrievalExperiment(config, client, embedding_client)
result = experiment.run()
experiment.plot("my_plot.png")
```

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

### "Module not found" errors
```bash
pip install -r requirements.txt
```

Or install specific packages:
```bash
pip install openai numpy scipy matplotlib
```

### "Rate limit exceeded"
The script respects OpenAI rate limits. If you hit limits:
1. Wait a few minutes and retry
2. Use a higher-tier API key
3. Reduce num_trials in the script

### "Timeout" errors
Increase timeout in config:
```python
config.model.api_timeout = 120  # Default is 60
```

### Memory errors
The extended context experiment generates very long texts. If you run out of memory:
1. Reduce `context_lengths` to max 8000
2. Reduce `num_samples_per_length` from 10 to 5
3. Run on a machine with more RAM

## Custom Usage

### Run Single Experiment

```python
from query_drift.experiments import StressTestRetrievalExperiment
from query_drift.config import ExperimentConfig
from openai import OpenAI
from query_drift.utils.embeddings import EmbeddingClient

client = OpenAI()
embedding_client = EmbeddingClient(client)
config = ExperimentConfig()

experiment = StressTestRetrievalExperiment(config, client, embedding_client)
result = experiment.run()
experiment.plot("stress_test.png")

print(f"Ceiling broken: {result.metrics['ceiling_broken']}")
print(f"OU decay rate: {result.metrics['ou_decay_rate']}")
```

### Modify Experiment Parameters

```python
# Reduce distractor counts for faster testing
class QuickStressTest(StressTestRetrievalExperiment):
    def run(self):
        distractor_counts = [50, 100, 200]  # Fewer levels
        num_trials = 2                       # Fewer trials
        # ... rest of implementation
```

### Save Custom Outputs

```python
result = experiment.run()

# Save to custom location
import json
with open("my_results.json", "w") as f:
    json.dump(result.metrics, f, indent=2)

# Custom plotting
experiment.plot("my_custom_plot.png")
```

## Understanding the Results

### Key Metrics

**Stress Test Retrieval:**
- `ceiling_broken`: True if max accuracy < 99%
- `ou_decay_rate`: Rate parameter k in exp(-k·x)
- `ou_baseline`: Asymptotic baseline accuracy
- `ou_r_squared`: Fit quality (> 0.9 is excellent)

**Entropy Drift:**
- `saturation_detected`: True if entropy plateaus
- `better_model`: "saturation" (OU) or "linear" (Brownian)
- `saturation_r_squared` vs `linear_r_squared`: Compare fits
- Higher R² wins

**Extended Context:**
- `max_context_length`: Maximum tested (should be 16000)
- `saturation_detected`: True if variance plateaus
- `better_model`: Which model fits better
- `extends_beyond_training`: Confirms > 4k tokens tested

**Enhanced Embedding Drift:**
- `mean_hurst`: Mean H across 5 trials
- `std_hurst`: Standard deviation
- `prediction_validated`: True if |H - 0.5| < 0.15
- Should be ~0.5 for Brownian motion

### Interpreting Plots

**Stress Test Plot:**
- X-axis: Number of distractors
- Y-axis: Retrieval accuracy
- Red dashed line: OU exponential fit
- Should show exponential decay to baseline

**Entropy Plot:**
- X-axis: Token position
- Y-axis: Output entropy H(p)
- Green line: Saturation (OU) model
- Red line: Linear (Brownian) model
- Green should fit better

**Extended Context Plot:**
- X-axis: Context length (tokens)
- Y-axis: Embedding variance
- Green line: Saturation model with plateau
- Should show variance leveling off

**Embedding Drift Plots:**
- X-axis: Continuation position (log scale)
- Y-axis: Variance (log scale)
- Blue line: Power law fit σ² ∝ t^(2H)
- Red line: Theoretical H=0.5
- Should overlap closely

## Next Steps

1. **Review summary report** for pass/fail status
2. **Check R² values** - should be > 0.85 for good fits
3. **Verify saturation** - better_model should be "saturation" for experiments 2 & 3
4. **Update paper** with new experimental results
5. **Include plots** in paper figures
6. **Document methodology** in paper methods section

## Support

For issues or questions:
1. Check existing results in `paper_experiments/` directory
2. Review detailed documentation in `BULLETPROOF_EXPERIMENTS.md`
3. Examine experiment source code in `query_drift/experiments/`

## Citation

When using these results in your paper, cite the enhanced methodology:

```
We conducted bulletproof validation experiments addressing major critiques:
(1) Stress tests with up to 1,000 distractors to break the ceiling effect,
(2) Entropy measurements controlling for LayerNorm artifacts,
(3) Extended context tests up to 16,000 tokens beyond typical training windows,
(4) Enhanced statistical rigor with 5 independent trials per configuration.
```
