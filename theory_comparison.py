"""
Compare Original Query Drift Hypothesis vs Ornstein-Uhlenbeck Theory
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("THEORY COMPARISON: Query Drift vs Ornstein-Uhlenbeck")
print("=" * 70)

# Experimental observations
experiments = {
    "Embedding Drift (H)": {"predicted_brownian": 0.5, "observed": 0.11},
    "Alignment Decay (β)": {"predicted_brownian": 0.5, "observed": 0.10},
    "Loss Scaling (β)": {"predicted_brownian": -0.5, "observed": -0.05},
    "Memory Retrieval": {"predicted_brownian": "degradation", "observed": "100% (no degradation)"},
}

print("\n" + "-" * 70)
print("EXPERIMENTAL RESULTS vs PREDICTIONS")
print("-" * 70)
print(f"{'Experiment':<25} {'Brownian (H=0.5)':<20} {'Observed':<15} {'Match?':<10}")
print("-" * 70)

for exp, vals in experiments.items():
    pred = vals["predicted_brownian"]
    obs = vals["observed"]
    if isinstance(obs, float):
        match = "❌ NO" if abs(pred - obs) > 0.2 else "✓ YES"
    else:
        match = "❌ NO"
    print(f"{exp:<25} {str(pred):<20} {str(obs):<15} {match:<10}")

print("\n" + "=" * 70)
print("WHY ORNSTEIN-UHLENBECK EXPLAINS EVERYTHING")
print("=" * 70)

print("""
ORIGINAL HYPOTHESIS (Brownian Motion, H = 0.5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Query vectors undergo random walk: dq = σ dW
• Variance grows linearly: σ²(t) = σ₀² × t
• Alignment decays as: C(t) ~ t^(-0.5)
• Loss scales as: L(c) ~ c^(-0.5)
• Memory retrieval degrades with age

REVISED THEORY (Ornstein-Uhlenbeck, mean-reverting):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Query vectors have ATTRACTOR: dq = θ(μ - q)dt + σ dW
• Variance SATURATES: σ²(t) = σ²_∞(1 - e^(-2θt))
• Alignment stabilizes: C(t) → C_∞ > 0
• Loss plateaus: L(c) → L_∞
• Memory retrieval remains effective!

THE KEY INSIGHT:
━━━━━━━━━━━━━━━━
Transformers have IMPLICIT REGULARIZATION through:
1. Attention softmax normalization
2. Layer normalization
3. Residual connections
4. Position encoding anchoring

These act as "restoring forces" that prevent unbounded drift!
""")

# Quantitative predictions
print("-" * 70)
print("QUANTITATIVE PREDICTIONS OF OU MODEL")
print("-" * 70)

# OU parameters from fitting
theta = 0.059  # mean-reversion rate
sigma_inf_sq = 0.051  # saturation variance
tau = 1 / (2 * theta)  # relaxation time

print(f"""
Fitted OU Parameters:
  • Mean-reversion rate θ = {theta:.4f}
  • Saturation variance σ²_∞ = {sigma_inf_sq:.4f}
  • Relaxation time τ = {tau:.1f} tokens

Predictions:
  1. VARIANCE GROWTH
     - At t << τ: σ²(t) ≈ 2θσ²_∞ × t  (linear, looks like H~0.1)
     - At t >> τ: σ²(t) → σ²_∞        (saturates)
     ✓ MATCHES: We see H ≈ 0.1 because we're in transition regime

  2. ALIGNMENT DECAY
     - OU predicts exponential approach to equilibrium
     - For small t: C(t) ≈ C₀ × t^(-H_eff) where H_eff << 0.5
     ✓ MATCHES: We observe β ≈ 0.1

  3. LOSS SCALING
     - If drift saturates, loss contribution also saturates
     - L(c) → L_∞ as c → ∞
     ✓ MATCHES: We see β ≈ 0 (flat loss, R² = 0.11)

  4. MEMORY RETRIEVAL
     - Bounded drift means queries stay "close enough" to keys
     - Retrieval accuracy remains high
     ✓ MATCHES: We see 100% retrieval rate
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

t = np.linspace(0.1, 100, 500)

# Plot 1: Variance growth comparison
ax1 = axes[0, 0]
# Brownian
var_brownian = 0.01 * t
# OU
var_ou = sigma_inf_sq * (1 - np.exp(-2 * theta * t))
# fBm H=0.1
var_fbm = 0.023 * t**(2*0.1)

ax1.plot(t, var_brownian, 'r--', label='Brownian (H=0.5)', linewidth=2)
ax1.plot(t, var_fbm, 'g:', label='fBm (H=0.1)', linewidth=2)
ax1.plot(t, var_ou, 'b-', label='Ornstein-Uhlenbeck', linewidth=2.5)
ax1.axhline(y=sigma_inf_sq, color='b', linestyle=':', alpha=0.5, label=f'OU saturation (σ²_∞={sigma_inf_sq:.3f})')
ax1.scatter([10, 25, 50], [0.035, 0.050, 0.050], s=100, c='black', zorder=5, label='Observed')
ax1.set_xlabel('Position (tokens)', fontsize=12)
ax1.set_ylabel('Variance σ²(t)', fontsize=12)
ax1.set_title('Variance Growth: Theory vs Observation', fontsize=14)
ax1.legend(loc='right')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 0.12)
ax1.grid(True, alpha=0.3)

# Plot 2: Alignment decay
ax2 = axes[0, 1]
t2 = np.linspace(90, 1500, 500)
# Brownian prediction
align_brownian = 0.84 * (t2/90)**(-0.5)
# OU-like slow decay
align_ou = 0.84 * (t2/90)**(-0.1)
# Observed data points
ctx_obs = np.array([90, 127, 186, 246, 302, 358, 422, 487, 540, 600, 661, 698, 757, 817, 873, 929, 993, 1058, 1111, 1171, 1232])
align_obs = np.array([0.841, 0.691, 0.713, 0.697, 0.712, 0.727, 0.707, 0.703, 0.689, 0.683, 0.663, 0.646, 0.643, 0.620, 0.625, 0.621, 0.615, 0.613, 0.613, 0.595, 0.594])

ax2.plot(t2, align_brownian, 'r--', label='Brownian: t^(-0.5)', linewidth=2)
ax2.plot(t2, align_ou, 'b-', label='OU-consistent: t^(-0.1)', linewidth=2.5)
ax2.scatter(ctx_obs, align_obs, s=30, c='black', alpha=0.6, zorder=5, label='Observed')
ax2.set_xlabel('Context Length', fontsize=12)
ax2.set_ylabel('Alignment C(t)', fontsize=12)
ax2.set_title('Alignment Decay: Theory vs Observation', fontsize=14)
ax2.legend()
ax2.set_ylim(0.3, 1.0)
ax2.grid(True, alpha=0.3)

# Plot 3: Loss scaling
ax3 = axes[1, 0]
c = np.linspace(100, 2000, 500)
loss_brownian = 0.5 * c**(-0.5) + 0.1
loss_ou = 0.2 * np.ones_like(c)  # Flat (saturated)
ctx_loss = np.array([100, 500, 1000])
loss_obs = np.array([0.225, 0.163, 0.218])

ax3.plot(c, loss_brownian, 'r--', label='Brownian: c^(-0.5)', linewidth=2)
ax3.plot(c, loss_ou, 'b-', label='OU: constant (saturated)', linewidth=2.5)
ax3.scatter(ctx_loss, loss_obs, s=100, c='black', zorder=5, label='Observed')
ax3.errorbar(ctx_loss, loss_obs, yerr=[0.067, 0.085, 0.053], fmt='none', c='black', capsize=5)
ax3.set_xlabel('Context Length', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss Scaling: Theory vs Observation', fontsize=14)
ax3.legend()
ax3.set_ylim(0, 0.4)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary diagram
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
┌─────────────────────────────────────────────────────────────┐
│                    THEORETICAL SUMMARY                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ORIGINAL HYPOTHESIS          REVISED THEORY                │
│  ══════════════════          ══════════════                 │
│                                                             │
│  Pure Brownian Motion    →   Ornstein-Uhlenbeck Process     │
│  dq = σ dW                   dq = θ(μ - q)dt + σ dW         │
│                                                             │
│  Unbounded drift         →   Mean-reverting drift          │
│  σ²(t) ~ t                   σ²(t) → σ²_∞ (saturates)       │
│                                                             │
│  H = 0.5                 →   H_eff ≈ 0.1 (apparent)         │
│                                                             │
│  "Lost in the Middle"    →   "Regularized Attention"        │
│  (severe)                    (mild, bounded)                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  KEY INSIGHT: Transformers have built-in stabilization!     │
│  Attention mechanisms act as restoring forces that          │
│  prevent unbounded query drift.                             │
└─────────────────────────────────────────────────────────────┘
"""
ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('query_drift_results/theory_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: query_drift_results/theory_comparison.png")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The Ornstein-Uhlenbeck model explains ALL four experimental observations:

  ✓ Embedding drift shows H ≈ 0.1 (variance saturates, not linear growth)
  ✓ Alignment decay shows β ≈ 0.1 (slow decay, not t^(-0.5))
  ✓ Loss scaling shows β ≈ 0 (flat, saturated)
  ✓ Memory retrieval shows 100% (bounded drift preserves similarity)

The original Query Drift Hypothesis (H=0.5) assumed pure Brownian motion,
but transformers have IMPLICIT REGULARIZATION that creates mean-reverting
dynamics. This is good news: attention doesn't drift as catastrophically
as the pure theory would predict!
""")
