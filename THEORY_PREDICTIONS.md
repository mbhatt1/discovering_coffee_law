# What Does the COFFEE Law Theory Actually Predict?

## The Core Theory

**COFFEE Law Claim**: Query vectors in transformers follow an **Ornstein-Uhlenbeck (OU) process** with **variance saturation**, not unbounded Brownian drift.

## Mathematical Predictions

### **OU Process (COFFEE Law)**
```
dq = -θ·q·dt + σ·dW

Key property: Variance saturates to σ²/(2θ)
```

### **Brownian Motion (Alternative)**
```
dq = σ·dW

Key property: Variance grows unboundedly as σ²·t
```

## What This Means for Retrieval

### **Prediction 1: Exponential Decay to Plateau (OU)**

The theory predicts retrieval metrics follow:
```
accuracy(n) = A · exp(-k·n) + baseline
```

Where:
- `A` = initial performance above baseline
- `k` = decay rate (related to θ in OU process)
- `baseline` = equilibrium performance level (the saturation point)
- `n` = number of distractors

**Key insight**: Performance **stabilizes** at `baseline`, doesn't keep declining.

### **Counter-prediction: Unbounded Decline (Brownian)**

If Brownian motion were correct:
```
accuracy(n) ~ 1 / sqrt(n)  or  accuracy(n) ~ n^(-β) with β > 0
```

Performance keeps declining without limit as n increases.

## What the Experiments Should Show

### **Scenario A: Baseline = 50-70%** (Moderate Robustness)
```
50 distractors:     95% accuracy
1000 distractors:   70% accuracy  
100000 distractors: 62% accuracy ← PLATEAU
```

✅ **Validates OU**: Exponential decay to 62% baseline  
❌ **Refutes Brownian**: If Brownian, accuracy at 100k << 62%

### **Scenario B: Baseline = 90-95%** (High Robustness)
```
50 distractors:     98% accuracy
1000 distractors:   94% accuracy
100000 distractors: 92% accuracy ← PLATEAU
```

✅ **Validates OU**: Exponential decay to 92% baseline  
❌ **Refutes Brownian**: If Brownian, accuracy would keep declining

### **Scenario C: Baseline ≈ 100%** (Extreme Robustness)
```
50 distractors:     100% accuracy
1000 distractors:   99.5% accuracy
100000 distractors: 99.2% accuracy ← PLATEAU (but at very high level)
```

**This is trickier but still validates OU IF**:
- MRR shows degradation: 0.98 → 0.85
- Mean rank grows: 1.0 → 1.5
- Variance in retrieval scores increases then stabilizes

## The Critical Distinction

### **What DOESN'T Validate the Theory**
If you see LINEAR decline:
```
accuracy(n) = 1 - c·n  (straight line downward)
```
This suggests Brownian motion, not OU.

### **What DOES Validate the Theory**
If you see EXPONENTIAL SATURATION:
```
accuracy(n) = 0.6 + 0.4·exp(-0.001·n)
```
This confirms OU process regardless of where it saturates (60%, 90%, or 99%).

## Why High Accuracy Doesn't Contradict the Theory

**Common Misconception**: "The theory predicts retrieval fails"

**Actual Theory**: "The theory predicts variance saturates to a bounded value"

**Implication**: Saturation can occur at ANY equilibrium level:
- **Low baseline** (50%): Query variance saturated after significant drift
- **Medium baseline** (70%): Query variance saturated with moderate drift  
- **High baseline** (95%): Query variance saturated with minimal drift

All three scenarios validate OU over Brownian!

## What We're Actually Testing

### **Primary Question**
Does performance follow:
- ✅ **OU**: Exponential saturation `y = A·exp(-k·x) + b`
- ❌ **Brownian**: Power law decline `y = A·x^(-β)`

### **Secondary Metrics** (if accuracy saturates too high)
- **MRR (Mean Reciprocal Rank)**: More sensitive, should still show exponential decay
- **Mean Rank**: Should grow then plateau
- **Rank Variance**: Should increase then stabilize

## Concrete Test Cases

### **Test 1: R² Comparison**
Fit both models to data:
```python
ou_model:      y = A·exp(-k·x) + b  
brownian_model: y = A·x^(-β)
```

**Expected**: `R²(OU) > R²(Brownian)` validates COFFEE Law

### **Test 2: Plateau Detection**
Compare first half vs second half growth:
```python
first_half_decay  = accuracy[0] - accuracy[mid]
second_half_decay = accuracy[mid] - accuracy[-1]
```

**Expected OU**: `second_half_decay < first_half_decay` (slowing decay)  
**Expected Brownian**: `second_half_decay ≈ first_half_decay` (constant rate)

### **Test 3: Variance Metrics**
Even if accuracy stays high, measure:
```python
- Rank variance across queries
- MRR coefficient of variation
- Embedding distance variance
```

**Expected OU**: These saturate (stop growing)  
**Expected Brownian**: These keep increasing

## Summary Table

| Outcome | Accuracy @ 100k | MRR @ 100k | Theory Validated? |
|---------|----------------|------------|-------------------|
| A | 60% (plateau) | 0.45 | ✅ OU (low baseline) |
| B | 90% (plateau) | 0.88 | ✅ OU (high baseline) |
| C | 98% (plateau) | 0.94 | ✅ OU (very high baseline) |
| D | 40% (declining) | 0.20 | ❌ Brownian (no plateau) |

**Key insight**: Outcomes A, B, and C **all validate** the COFFEE Law. Only outcome D would refute it.

## Bottom Line

**The COFFEE Law predicts**: System reaches equilibrium (variance saturation)

**It does NOT predict**: Retrieval must fail

**The test is**: Does performance plateau (OU) or decline indefinitely (Brownian)?

**High accuracy at 100k distractors** means equilibrium happened at a high-performance level, which **supports** the theory, doesn't contradict it.
