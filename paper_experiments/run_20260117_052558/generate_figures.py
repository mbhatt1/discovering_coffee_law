#!/usr/bin/env python3
"""Generate publication-quality figures for the paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open("master_results.json") as f:
    results = json.load(f)

# Create figures directory
Path("../figures").mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# =============================================================================
# Figure 1: Model Comparison (Brownian vs fBm vs OU)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Get variance data
vg_trials = results["section3"]["3.1_variance_growth"]["trials"]
all_positions = []
all_variances = []
for trial in vg_trials:
    all_positions.extend(trial["positions"])
    all_variances.extend(trial["variances"])

positions = np.array(all_positions)
variances = np.array(all_variances)

# Unique positions for plotting
unique_pos = np.array(sorted(set(positions)))
mean_var = np.array([np.mean(variances[positions == p]) for p in unique_pos])

t_smooth = np.linspace(1, 120, 200)

# Panel A: Brownian
ax = axes[0]
ax.scatter(unique_pos, mean_var, s=80, c='blue', label='Data', zorder=3)
# Brownian fit
A_brown = 0.00113
var_brown = A_brown * t_smooth
ax.plot(t_smooth, var_brown, 'r--', lw=2, label=f'Brownian (R²=-45)')
ax.set_xlabel('Position (tokens)')
ax.set_ylabel('Variance σ²(t)')
ax.set_title('A) Standard Brownian Motion')
ax.legend()
ax.set_ylim(0, 0.15)

# Panel B: fBm
ax = axes[1]
ax.scatter(unique_pos, mean_var, s=80, c='blue', label='Data', zorder=3)
# fBm fit
A_fbm = 0.057
H_fbm = 0.037
var_fbm = A_fbm * np.power(t_smooth, 2*H_fbm)
ax.plot(t_smooth, var_fbm, 'g--', lw=2, label=f'fBm H={H_fbm:.3f} (R²=0.60)')
ax.set_xlabel('Position (tokens)')
ax.set_ylabel('Variance σ²(t)')
ax.set_title('B) Fractional Brownian Motion')
ax.legend()
ax.set_ylim(0, 0.15)

# Panel C: OU
ax = axes[2]
ax.scatter(unique_pos, mean_var, s=80, c='blue', label='Data', zorder=3)
# OU fit
sigma_inf = 0.078
theta = 0.083
var_ou = sigma_inf * (1 - np.exp(-2 * theta * t_smooth))
ax.plot(t_smooth, var_ou, 'purple', lw=2, label=f'OU θ={theta:.3f} (R²=0.86)')
ax.axhline(y=sigma_inf, color='purple', ls=':', alpha=0.5, label=f'σ²_∞={sigma_inf:.3f}')
ax.set_xlabel('Position (tokens)')
ax.set_ylabel('Variance σ²(t)')
ax.set_title('C) Ornstein-Uhlenbeck')
ax.legend()
ax.set_ylim(0, 0.15)

plt.tight_layout()
plt.savefig('../figures/fig1_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig1_model_comparison.pdf', bbox_inches='tight')
print("Saved: fig1_model_comparison.png/pdf")

# =============================================================================
# Figure 2: Alignment Decay
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ad_trial = results["section3"]["3.2_alignment_decay"]["trials"][0]
ctx = np.array(ad_trial["context_lengths"])
align = np.array(ad_trial["alignments"])

ax.scatter(ctx, align, s=40, c='blue', alpha=0.7, label='Observed')

# Fit lines
t_fit = np.linspace(ctx.min(), ctx.max(), 200)
# Observed fit
A_obs = 0.873
beta_obs = 0.166
align_fit = A_obs * np.power(t_fit, -beta_obs)
ax.plot(t_fit, align_fit, 'b-', lw=2, label=f'Fit: C(t) ∝ t^(-{beta_obs:.2f})')

# Theoretical
A_theory = align[0] * ctx[0]**0.5
align_theory = A_theory * np.power(t_fit, -0.5)
ax.plot(t_fit, align_theory, 'r--', lw=2, alpha=0.7, label='Theory: C(t) ∝ t^(-0.5)')

ax.set_xlabel('Context Length (characters)')
ax.set_ylabel('Alignment C(t) = ⟨q_t, u⟩/||q_t||')
ax.set_title('Alignment Decay: Observed vs Theoretical')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig2_alignment_decay.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig2_alignment_decay.pdf', bbox_inches='tight')
print("Saved: fig2_alignment_decay.png/pdf")

# =============================================================================
# Figure 3: Temperature Dependence
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

temp_data = results["section5"]["5.1_temperature"]["by_temperature"]
temps = []
h_values = []
r2_values = []

for temp, data in sorted(temp_data.items(), key=lambda x: float(x[0])):
    temps.append(float(temp))
    h_values.append(data["hurst_exponent"])
    r2_values.append(data["r_squared"])

temps = np.array(temps)
h_values = np.array(h_values)
r2_values = np.array(r2_values)

# Panel A: H vs Temperature
ax = axes[0]
ax.scatter(temps, h_values, s=100, c='blue', zorder=3)
ax.axhline(y=0.5, color='red', ls='--', label='Brownian (H=0.5)')
ax.axhline(y=0.13, color='green', ls=':', label='Mean H (T>0) ≈ 0.13')
ax.set_xlabel('Temperature')
ax.set_ylabel('Hurst Exponent H')
ax.set_title('A) Hurst Exponent vs Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: R² vs Temperature
ax = axes[1]
colors = ['red' if r < 0 else 'blue' for r in r2_values]
ax.scatter(temps, r2_values, s=100, c=colors, zorder=3)
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Temperature')
ax.set_ylabel('R² (fit quality)')
ax.set_title('B) Fit Quality vs Temperature')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/fig3_temperature.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig3_temperature.pdf', bbox_inches='tight')
print("Saved: fig3_temperature.png/pdf")

# =============================================================================
# Figure 4: Domain Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

domain_data = results["section5"]["5.3_domain"]["by_domain"]
domains = list(domain_data.keys())
variances = [domain_data[d]["variance"] for d in domains]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(domains)))
bars = ax.bar(domains, variances, color=colors, edgecolor='black')

ax.set_xlabel('Text Domain')
ax.set_ylabel('Embedding Variance')
ax.set_title('Variance by Text Domain')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, var in zip(bars, variances):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{var:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../figures/fig4_domains.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig4_domains.pdf', bbox_inches='tight')
print("Saved: fig4_domains.png/pdf")

# =============================================================================
# Figure 5: Summary Schematic
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary = """
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                   ORNSTEIN-UHLENBECK DYNAMICS IN TRANSFORMERS                    │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   STOCHASTIC EQUATION:    dq = θ(μ - q)dt + σ dW                                 │
│                                ↑                                                 │
│                         Restoring Force                                          │
│                                                                                  │
│   FITTED PARAMETERS:                                                             │
│       • Mean-reversion rate:  θ = 0.083                                          │
│       • Saturation variance:  σ²_∞ = 0.078                                       │
│       • Relaxation time:      τ = 6 tokens                                       │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   KEY FINDINGS:                                                                  │
│                                                                                  │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│   │ VARIANCE GROWTH │    │ ALIGNMENT DECAY │    │ MEMORY RETRIEVAL│             │
│   │                 │    │                 │    │                 │             │
│   │ H = 0.04        │    │ β = 0.17        │    │ Rate = 100%     │             │
│   │ (vs 0.5 theory) │    │ (vs 0.5 theory) │    │ (no degradation)│             │
│   │                 │    │                 │    │                 │             │
│   │ 12× SLOWER      │    │ 3× SLOWER       │    │ PRESERVED       │             │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘             │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   IMPLICATIONS:                                                                  │
│       • Transformer attention is MORE STABLE than Brownian theory predicts       │
│       • "Lost in the Middle" effect is WEAK or ABSENT                            │
│       • Attention mechanisms act as implicit RESTORING FORCES                    │
│       • Context can be longer before significant degradation                     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.5, 0.5, summary, transform=ax.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.savefig('../figures/fig5_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig5_summary.pdf', bbox_inches='tight')
print("Saved: fig5_summary.png/pdf")

print("\nAll figures generated successfully!")
