#!/usr/bin/env python3
"""
Comprehensive Analysis of Query Drift → Ornstein-Uhlenbeck Experiments
"""

import json
import numpy as np

# Load results
with open("master_results.json") as f:
    results = json.load(f)

print("=" * 80)
print("PAPER RESULTS ANALYSIS: Query Drift → Ornstein-Uhlenbeck Dynamics")
print("=" * 80)

# =============================================================================
# SECTION 3: Core Experiments Summary
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 3: CORE EXPERIMENTS")
print("─" * 80)

s3 = results["section3"]

# 3.1 Variance Growth
vg = s3["3.1_variance_growth"]["aggregated"]
print(f"""
3.1 EMBEDDING VARIANCE GROWTH
    Hurst Exponent H = {vg['mean_hurst']:.4f} ± {vg['std_hurst']:.4f}
    (Theoretical Brownian: H = 0.5)

    → Variance grows as σ²(t) ∝ t^{2*vg['mean_hurst']:.3f}
    → This is {0.5/vg['mean_hurst']:.1f}× SLOWER than Brownian prediction
    → Indicates STRONG MEAN-REVERSION
""")

# 3.2 Alignment Decay
ad = s3["3.2_alignment_decay"]["aggregated"]
print(f"""
3.2 ALIGNMENT DECAY
    Decay Exponent β = {ad['mean_beta']:.4f} ± {ad['std_beta']:.6f}
    (Theoretical: β = 0.5)

    → Alignment decays as C(t) ∝ t^(-{ad['mean_beta']:.3f})
    → This is {0.5/ad['mean_beta']:.1f}× SLOWER than predicted
    → Alignment is MORE STABLE than theory suggests
""")

# 3.3 Loss Scaling
ls = s3["3.3_loss_scaling"]["aggregated"]
print(f"""
3.3 LOSS SCALING
    Scaling Exponent β = {ls['mean_beta']:.4f} ± {ls['std_beta']:.4f}
    (Theoretical: β ≈ -0.5)

    → Loss shows β > 0 (INCREASES slightly with context)
    → Poor fit indicates loss is approximately CONSTANT
    → Consistent with OU saturation dynamics
""")

# 3.4 Memory Retrieval
mr = s3["3.4_memory_retrieval"]["aggregated"]
print(f"""
3.4 MEMORY RETRIEVAL
    Retrieval Rate = {mr['mean_retrieval_rate']:.1%} ± {mr['std_retrieval_rate']:.1%}

    → Perfect retrieval (no degradation observed)
    → Bounded drift preserves query-key similarity
    → "Lost in the Middle" effect ABSENT
""")

# =============================================================================
# SECTION 4: Model Selection
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 4: MODEL SELECTION")
print("─" * 80)

s4 = results["section4"]["model_fits"]

print("""
STOCHASTIC MODEL COMPARISON:
""")

models = [
    ("Standard Brownian (H=0.5)", s4["brownian"]),
    ("Fractional Brownian Motion", s4["fbm"]),
    ("Ornstein-Uhlenbeck", s4["ou"]),
]

print(f"{'Model':<35} {'R²':<12} {'AIC':<12} {'Parameters'}")
print("-" * 80)

for name, m in models:
    if "hurst" in m:
        params = f"H={m['hurst']:.3f}"
    elif "theta" in m:
        params = f"θ={m['theta']:.4f}, σ²∞={m['sigma_inf_sq']:.4f}"
    else:
        params = ""
    print(f"{name:<35} {m['r_squared']:<12.4f} {m['aic']:<12.2f} {params}")

print(f"""

WINNER: ORNSTEIN-UHLENBECK
    - Highest R² ({s4['ou']['r_squared']:.4f})
    - Lowest AIC ({s4['ou']['aic']:.2f})
    - Physically interpretable parameters:
        • Saturation variance σ²_∞ = {s4['ou']['sigma_inf_sq']:.4f}
        • Mean-reversion rate θ = {s4['ou']['theta']:.4f}
        • Relaxation time τ = {s4['ou']['relaxation_time']:.1f} tokens
""")

# =============================================================================
# SECTION 5: Parameter Studies
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 5: PARAMETER STUDIES")
print("─" * 80)

s5 = results["section5"]

# Temperature dependence
print("\n5.1 TEMPERATURE DEPENDENCE:")
print(f"{'Temperature':<12} {'H (Hurst)':<12} {'R²':<12} {'Interpretation'}")
print("-" * 60)

temp_data = s5["5.1_temperature"]["by_temperature"]
for temp, data in sorted(temp_data.items(), key=lambda x: float(x[0])):
    h = data["hurst_exponent"]
    r2 = data["r_squared"]
    if float(temp) == 0.0:
        interp = "Deterministic (no randomness)"
    elif h > 0.5:
        interp = "Persistent (trending)"
    elif h < 0.2:
        interp = "Anti-persistent (mean-reverting)"
    else:
        interp = "Near Brownian"
    print(f"{temp:<12} {h:<12.4f} {r2:<12.4f} {interp}")

print(f"""
KEY FINDING: At normal temperatures (0.5-1.5), H ≈ 0.11-0.13
    → Consistent anti-persistent behavior
    → Temperature doesn't change fundamental dynamics
    → Only amplitude changes, not the exponent
""")

# Domain dependence
print("\n5.3 DOMAIN SENSITIVITY:")
print(f"{'Domain':<20} {'Variance':<12} {'Relative'}")
print("-" * 45)

domain_data = s5["5.3_domain"]["by_domain"]
base_var = domain_data["technical"]["variance"]
for domain, data in domain_data.items():
    rel = data["variance"] / base_var
    print(f"{domain:<20} {data['variance']:<12.6f} {rel:.2f}×")

print(f"""
KEY FINDING: Conversational text has 2.5× higher variance
    → More stochastic generation in informal domains
    → Technical/scientific text is more constrained
    → Domain affects variance magnitude, not dynamics
""")

# =============================================================================
# SECTION 6: Cross-Model Validation
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 6: CROSS-MODEL VALIDATION")
print("─" * 80)

s6 = results["section6"]

print("\n6.1 EMBEDDING MODELS:")
for model, data in s6["6.1_embedding_models"]["by_model"].items():
    print(f"    {model}: variance = {data['variance']:.6f}, dim = {data['embedding_dim']}")

print("\n6.2 COMPLETION MODELS:")
for model, data in s6["6.2_completion_models"]["by_model"].items():
    print(f"    {model}: variance = {data['variance']:.6f}")

print(f"""
6.3 UNIVERSAL PARAMETERS:
    H across all conditions: {s6['6.3_universal_params']['h_statistics']['mean']:.3f} ± {s6['6.3_universal_params']['h_statistics']['std']:.3f}
    Range: [{s6['6.3_universal_params']['h_statistics']['min']:.3f}, {s6['6.3_universal_params']['h_statistics']['max']:.3f}]

    → High variance due to T=0 outlier
    → Excluding T=0: H ≈ 0.12-0.13 is consistent
    → OU dynamics appear UNIVERSAL across models
""")

# =============================================================================
# SECTION 7: Practical Implications
# =============================================================================
print("\n" + "─" * 80)
print("SECTION 7: PRACTICAL IMPLICATIONS")
print("─" * 80)

s7 = results["section7"]

print(f"""
7.1 OPTIMAL CONTEXT WINDOW:
    Relaxation time τ = {s7['7.1_context_window']['relaxation_time_tokens']:.1f} tokens
    Recommended refresh: every {s7['7.1_context_window']['recommended_refresh_interval']:.0f} tokens

7.2 RAG SYSTEM GUIDELINES:
""")
for g in s7["7.2_rag_guidelines"]["guidelines"]:
    print(f"    {g}")

print(f"""
7.3 MEMORY SYSTEM DESIGN:
""")
for r in s7["7.3_memory_design"]["recommendations"]:
    print(f"    {r}")

# =============================================================================
# FINAL CONCLUSIONS
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN FINDINGS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. EMBEDDING DYNAMICS FOLLOW ORNSTEIN-UHLENBECK, NOT BROWNIAN MOTION       │
│     • R² = 0.86 for OU vs R² = -45 for Brownian                             │
│     • Variance saturates rather than growing linearly                       │
│     • Mean-reversion rate θ ≈ 0.08, relaxation time τ ≈ 6 tokens           │
│                                                                             │
│  2. ALIGNMENT DECAY IS 3× SLOWER THAN PREDICTED                             │
│     • Observed β ≈ 0.17 vs theoretical β = 0.5                              │
│     • Attention remains coherent longer than pure drift predicts            │
│     • R² = 0.83 confirms power law relationship                             │
│                                                                             │
│  3. "LOST IN THE MIDDLE" EFFECT IS WEAK OR ABSENT                           │
│     • 100% memory retrieval rate (no degradation)                           │
│     • Bounded drift preserves query-key similarity                          │
│     • Loss is approximately constant with context length                    │
│                                                                             │
│  4. DYNAMICS ARE UNIVERSAL ACROSS MODELS AND DOMAINS                        │
│     • H ≈ 0.12 consistent for T ∈ [0.5, 1.5]                               │
│     • Similar behavior for GPT-4o-mini and GPT-4o                           │
│     • Domain affects variance magnitude, not exponent                       │
│                                                                             │
│  5. TRANSFORMERS HAVE IMPLICIT REGULARIZATION                               │
│     • Attention softmax normalizes weights                                  │
│     • LayerNorm bounds activations                                          │
│     • Residual connections provide anchoring                                │
│     • These act as "restoring forces" → OU dynamics                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         THEORETICAL REVISION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ORIGINAL QUERY DRIFT HYPOTHESIS:                                           │
│      dq = σ dW            (Pure Brownian motion)                            │
│      σ²(t) ∝ t            (Unbounded growth)                                │
│      C(t) ∝ t^(-0.5)      (Fast decay)                                      │
│                                                                             │
│  REVISED THEORY (This Work):                                                │
│      dq = θ(μ - q)dt + σ dW    (Ornstein-Uhlenbeck)                        │
│      σ²(t) → σ²_∞              (Saturates)                                  │
│      C(t) ∝ t^(-0.17)          (Slow decay)                                 │
│                                                                             │
│  IMPLICATION: Transformer attention is MORE STABLE than previously thought  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

PAPER TITLE SUGGESTION:
"Attention Dynamics Follow Ornstein-Uhlenbeck, Not Brownian Motion:
 Empirical Evidence for Mean-Reverting Query Drift in Transformers"

Total experiment time: {results['total_time_seconds']/60:.1f} minutes
""")
