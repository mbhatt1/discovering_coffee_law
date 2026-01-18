#!/usr/bin/env python3
"""
Generate all paper figures from latest experimental runs.
Fixed to match actual data structure from run_20260117_170920.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load latest experimental data
LATEST_RUN = Path('paper_experiments/run_20260117_170920')
OUTPUT_DIR = Path('paper')

with open(LATEST_RUN / 'master_results.json', 'r') as f:
    data = json.load(f)

# Also load bulletproof results
with open('bulletproof_results_20260117_172001/stress_test/results.json', 'r') as f:
    stress_data = json.load(f)

with open('bulletproof_results_20260117_172001/entropy_drift/results.json', 'r') as f:
    entropy_data = json.load(f)

print("Generating all paper figures from latest experimental data...")

# ============================================================================
# Figure 1: Model Comparison (Variance Growth)
# ============================================================================
def generate_fig1_model_comparison():
    """Generate model comparison showing variance growth."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract variance data from section 3 trials
    variance_data = data['section3']['3.1_variance_growth']
    trials = variance_data['trials']

    # Aggregate positions and variances from all trials
    positions = np.array(trials[0]['positions'])  # Same for all trials
    all_variances = [np.array(trial['variances']) for trial in trials]
    mean_var = np.mean(all_variances, axis=0)
    std_var = np.std(all_variances, axis=0)

    # Plot observed data
    ax.errorbar(positions, mean_var, yerr=std_var, fmt='o',
                label='Observed', capsize=3, markersize=6, color='black')

    # OU model fit from section4
    ou_params = data['section4']['model_fits']['ou']
    theta = ou_params['theta']
    sigma2_inf = ou_params['sigma_inf_sq']
    r_squared = ou_params['r_squared']

    t_smooth = np.linspace(1, max(positions), 100)
    ou_pred = sigma2_inf * (1 - np.exp(-2 * theta * t_smooth))
    ax.plot(t_smooth, ou_pred, '-', label=f'OU Model (R²={r_squared:.2f})',
            color='red', linewidth=2)

    # Brownian motion (linear)
    slope = mean_var[-1] / positions[-1]
    brownian_pred = slope * t_smooth
    ax.plot(t_smooth, brownian_pred, '--', label='Brownian Motion',
            color='gray', linewidth=2)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Variance')
    ax.set_title('Model Comparison: Variance Growth Over Token Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig1_model_comparison.pdf')
    plt.savefig(OUTPUT_DIR / 'fig1_model_comparison.png')
    plt.close()
    print("✓ Generated fig1_model_comparison")

# ============================================================================
# Figure 2: Alignment Decay
# ============================================================================
def generate_fig2_alignment_decay():
    """Generate alignment decay on log-log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    alignment_data = data['section3']['3.2_alignment_decay']

    # Use first trial (they're identical in this dataset)
    trial = alignment_data['trials'][0]
    lengths = np.array(trial['context_lengths'])
    alignments = np.array(trial['alignments'])

    # Log-log plot
    ax.loglog(lengths, alignments, 'o', alpha=0.6, markersize=5, label='Observed')

    # Power law fit
    beta = alignment_data['aggregated']['mean_beta']
    r_squared = trial['r_squared']

    # Fit line
    log_lengths = np.log(lengths)
    log_alignments = np.log(alignments)
    slope, intercept, _, _, _ = stats.linregress(log_lengths, log_alignments)

    t_smooth = np.logspace(np.log10(min(lengths)), np.log10(max(lengths)), 100)
    fit_line = np.exp(intercept) * t_smooth ** slope
    ax.loglog(t_smooth, fit_line, '-', linewidth=2, color='red',
              label=f'Fitted: $t^{{-{beta:.2f}}}$ (R²={r_squared:.2f})')

    # Brownian reference (β = 0.5)
    brownian_line = np.exp(intercept) * t_smooth ** (-0.5)
    ax.loglog(t_smooth, brownian_line, '--', linewidth=2, color='gray',
              label='Brownian: $t^{-0.5}$')

    ax.set_xlabel('Context Length (characters)')
    ax.set_ylabel('Alignment (cosine similarity)')
    ax.set_title('Alignment Decay (Log-Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.savefig(OUTPUT_DIR / 'fig2_alignment_decay.pdf')
    plt.savefig(OUTPUT_DIR / 'fig2_alignment_decay.png')
    plt.savefig(OUTPUT_DIR / 'alignment_power.png')  # Legacy name used in paper
    plt.close()
    print("✓ Generated fig2_alignment_decay")

# ============================================================================
# Figure 3: Temperature Dependence
# ============================================================================
def generate_fig3_temperature():
    """Generate temperature dependence plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    temp_data = data['section5']['5.1_temperature']['by_temperature']

    temperatures = sorted([float(t) for t in temp_data.keys()])
    hurst_exponents = [temp_data[str(t)]['hurst_exponent'] for t in temperatures]
    amplitudes = [temp_data[str(t)]['amplitude'] for t in temperatures]
    r_squared = [temp_data[str(t)]['r_squared'] for t in temperatures]

    # Left: Hurst exponent vs temperature
    ax1.plot(temperatures, hurst_exponents, 'o-', markersize=8, linewidth=2, color='blue')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Brownian (H=0.5)')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Hurst Exponent H')
    ax1.set_title('Hurst Exponent vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.6])

    # Right: R² vs temperature
    ax2.plot(temperatures, r_squared, 's-', markersize=8, linewidth=2, color='green')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('R² (Fit Quality)')
    ax2.set_title('Saturation Fit Quality vs Temperature')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_temperature.pdf')
    plt.savefig(OUTPUT_DIR / 'fig3_temperature.png')
    plt.close()
    print("✓ Generated fig3_temperature")

# ============================================================================
# Figure 4: Domain-Specific Patterns
# ============================================================================
def generate_fig4_domains():
    """Generate domain-specific variance comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    domain_data = data['section5']['5.3_domain']['by_domain']

    domains = list(domain_data.keys())
    variances = [domain_data[d]['variance'] for d in domains]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax.bar(domains, variances, color=colors[:len(domains)], alpha=0.8)

    ax.set_xlabel('Text Domain')
    ax.set_ylabel('Embedding Variance')
    ax.set_title('Variance by Text Domain')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, var in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{var:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_domains.pdf')
    plt.savefig(OUTPUT_DIR / 'fig4_domains.png')
    plt.close()
    print("✓ Generated fig4_domains")

# ============================================================================
# Figure 5: Summary
# ============================================================================
def generate_fig5_summary():
    """Generate 4-panel summary figure."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Top left: Variance saturation
    ax1 = fig.add_subplot(gs[0, 0])
    variance_data = data['section3']['3.1_variance_growth']
    trials = variance_data['trials']
    positions = np.array(trials[0]['positions'])
    all_variances = [np.array(trial['variances']) for trial in trials]
    mean_var = np.mean(all_variances, axis=0)
    std_var = np.std(all_variances, axis=0)

    ax1.errorbar(positions, mean_var, yerr=std_var, fmt='o-', markersize=6,
                 linewidth=2, capsize=3)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Variance')
    ax1.set_title('(A) Variance Saturation')
    ax1.grid(True, alpha=0.3)

    # Top right: Model comparison (R² values)
    ax2 = fig.add_subplot(gs[0, 1])
    model_fits = data['section4']['model_fits']
    models = ['OU\n(Saturating)', 'Fractional\nBrownian', 'Standard\nBrownian']
    r_squared_values = [
        model_fits['ou']['r_squared'],
        model_fits['fbm']['r_squared'],
        model_fits['brownian']['r_squared']
    ]
    colors = ['green', 'orange', 'red']
    bars = ax2.bar(models, r_squared_values, color=colors, alpha=0.7)
    ax2.set_ylabel('R² (Goodness of Fit)')
    ax2.set_title('(B) Model Comparison')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, r_squared_values):
        y_pos = max(val, 0) + 0.05 if val > 0 else val - 0.3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # Bottom left: Alignment decay
    ax3 = fig.add_subplot(gs[1, 0])
    alignment_data = data['section3']['3.2_alignment_decay']
    trial = alignment_data['trials'][0]
    lengths = np.array(trial['context_lengths'])
    alignments = np.array(trial['alignments'])
    beta = alignment_data['aggregated']['mean_beta']

    ax3.semilogx(lengths, alignments, 'o-', markersize=4, linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Context Length (characters)')
    ax3.set_ylabel('Alignment')
    ax3.set_title(f'(C) Alignment Decay: β = {beta:.2f}')
    ax3.grid(True, alpha=0.3, which='both')

    # Bottom right: Retrieval accuracy (from stress test)
    ax4 = fig.add_subplot(gs[1, 1])
    distractor_counts = stress_data['data']['distractor_counts']
    accuracies = stress_data['data']['mean_accuracies']

    ax4.semilogx(distractor_counts, [a*100 for a in accuracies], 'o-',
                 markersize=8, linewidth=2, color='green')
    ax4.set_xlabel('Number of Distractors')
    ax4.set_ylabel('Retrieval Accuracy (%)')
    ax4.set_title('(D) Retrieval Robustness')
    ax4.set_ylim([90, 102])
    ax4.axhline(y=95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax4.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig5_summary.pdf')
    plt.savefig(OUTPUT_DIR / 'fig5_summary.png')
    plt.close()
    print("✓ Generated fig5_summary")

# ============================================================================
# Figure 6: Variance Raw (NEW version)
# ============================================================================
def generate_fig6_variance_raw():
    """Generate variance growth with all trials."""
    fig, ax = plt.subplots(figsize=(8, 5))

    variance_data = data['section3']['3.1_variance_growth']
    trials = variance_data['trials']

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, trial in enumerate(trials):
        positions = np.array(trial['positions'])
        variances = np.array(trial['variances'])
        ax.plot(positions, variances, 'o-', color=colors[i], markersize=6,
                linewidth=2, alpha=0.8, label=f'Trial {i}')

    # Mean line
    all_variances = [np.array(trial['variances']) for trial in trials]
    mean_var = np.mean(all_variances, axis=0)
    ax.plot(positions, mean_var, 's--', color='black', markersize=8,
            linewidth=2.5, label='Mean')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Embedding Variance')
    ax.set_title('Variance Growth (3 Trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig_variance_NEW.png')
    plt.close()
    print("✓ Generated fig_variance_NEW.png")

# ============================================================================
# Figure 7: OU Prediction
# ============================================================================
def generate_fig7_ou_prediction():
    """Generate OU prediction validation plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    variance_data = data['section3']['3.1_variance_growth']
    trials = variance_data['trials']
    positions = np.array(trials[0]['positions'])
    all_variances = [np.array(trial['variances']) for trial in trials]
    mean_var = np.mean(all_variances, axis=0)
    std_var = np.std(all_variances, axis=0)

    ou_params = data['section4']['model_fits']['ou']
    theta = ou_params['theta']
    sigma2_inf = ou_params['sigma_inf_sq']

    # OU prediction
    ou_pred = sigma2_inf * (1 - np.exp(-2 * theta * positions))

    # Plot observed vs predicted
    ax.errorbar(positions, mean_var, yerr=std_var, fmt='o', markersize=8,
                label='Observed', color='black', capsize=3)

    t_smooth = np.linspace(1, max(positions), 100)
    ou_smooth = sigma2_inf * (1 - np.exp(-2 * theta * t_smooth))
    ax.plot(t_smooth, ou_smooth, '-', linewidth=2, label='OU Prediction', color='red')

    # Saturation line
    ax.axhline(y=sigma2_inf, color='red', linestyle=':', alpha=0.5,
               label=f'σ²∞ = {sigma2_inf:.3f}')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Variance')
    ax.set_title(f'OU Model Validation (θ={theta:.3f}, σ²∞={sigma2_inf:.3f}, τ={1/(2*theta):.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig7_ou_prediction.pdf')
    plt.savefig(OUTPUT_DIR / 'fig7_ou_prediction.png')
    plt.close()
    print("✓ Generated fig7_ou_prediction")

# ============================================================================
# Figure 8: Loss Scaling
# ============================================================================
def generate_fig8_loss_scaling():
    """Generate loss scaling plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    loss_data = data['section3']['3.3_loss_scaling']

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, trial in enumerate(loss_data['trials']):
        context_lengths = np.array(trial['context_lengths'])
        losses = np.array(trial['mean_losses'])
        ax.plot(context_lengths, losses, 'o-', color=colors[i], markersize=6,
                linewidth=2, alpha=0.8, label=f'Trial {i}')

    ax.set_xlabel('Context Length (characters)')
    ax.set_ylabel('Mean Loss (negative log-prob)')
    ax.set_title('Loss vs Context Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig8_loss_scaling.pdf')
    plt.savefig(OUTPUT_DIR / 'fig8_loss_scaling.png')
    plt.close()
    print("✓ Generated fig8_loss_scaling")

# ============================================================================
# Figure 9: Cross-Model Comparison
# ============================================================================
def generate_fig9_cross_model():
    """Generate cross-model comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Embedding model comparison
    embed_data = data['section6']['6.1_embedding_models']['by_model']
    embed_models = list(embed_data.keys())
    embed_variances = [embed_data[m]['variance'] for m in embed_models]
    embed_dims = [embed_data[m]['embedding_dim'] for m in embed_models]

    bars1 = ax1.bar(embed_models, embed_variances, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax1.set_xlabel('Embedding Model')
    ax1.set_ylabel('Variance')
    ax1.set_title('Embedding Model Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add dimension labels
    for bar, dim in zip(bars1, embed_dims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{dim}d', ha='center', va='bottom', fontsize=9)

    # Right: Completion model comparison
    comp_data = data['section6']['6.2_completion_models']['by_model']
    comp_models = list(comp_data.keys())
    comp_variances = [comp_data[m]['variance'] for m in comp_models]

    bars2 = ax2.bar(comp_models, comp_variances, color=['#F18F01', '#C73E1D'], alpha=0.8)
    ax2.set_xlabel('Completion Model')
    ax2.set_ylabel('Variance')
    ax2.set_title('Completion Model Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, var in zip(bars2, comp_variances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{var:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_cross_model.pdf')
    plt.savefig(OUTPUT_DIR / 'fig9_cross_model.png')
    plt.close()
    print("✓ Generated fig9_cross_model")

# ============================================================================
# Additional figures referenced in paper
# ============================================================================
def generate_stress_test_figure():
    """Generate stress test figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    distractor_counts = stress_data['data']['distractor_counts']
    accuracies = stress_data['data']['mean_accuracies']
    std_accs = stress_data['data']['std_accuracies']

    ax.errorbar(distractor_counts, [a*100 for a in accuracies],
                yerr=[s*100 for s in std_accs], fmt='o-',
                markersize=8, linewidth=2, capsize=3, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Distractors')
    ax.set_ylabel('Retrieval Accuracy (%)')
    ax.set_title('Stress Test: Accuracy vs Distractor Count')
    ax.set_ylim([90, 102])
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig_stress_test_NEW.png')
    plt.close()
    print("✓ Generated fig_stress_test_NEW.png")

def generate_entropy_control_figure():
    """Generate entropy control figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parse the string arrays from entropy data
    positions = np.array([10, 20, 30, 50, 70, 100, 130, 150])

    linear_fit = entropy_data['data']['linear_fit']
    sat_fit = entropy_data['data']['saturation_fit']

    # Generate fit lines
    t_smooth = np.linspace(10, 150, 100)
    linear_pred = linear_fit['slope'] * t_smooth + linear_fit['intercept']

    # Saturation: amplitude * (1 - exp(-rate * t)) + offset
    sat_pred = sat_fit['amplitude'] * (1 - np.exp(-sat_fit['rate'] * t_smooth)) + sat_fit['offset']

    ax.plot(t_smooth, linear_pred, '--', linewidth=2, color='gray',
            label=f'Linear (R²={linear_fit["r_squared"]:.2f})')
    ax.plot(t_smooth, sat_pred, '-', linewidth=2, color='red',
            label=f'Saturation (R²={sat_fit["r_squared"]:.2f})')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Output Entropy')
    ax.set_title('Entropy Control: Saturation vs Linear')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig_entropy_control_NEW.png')
    plt.close()
    print("✓ Generated fig_entropy_control_NEW.png")

def generate_variance_growth_loglog():
    """Generate variance growth on log-log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    variance_data = data['section3']['3.1_variance_growth']
    trials = variance_data['trials']
    positions = np.array(trials[0]['positions'])
    all_variances = [np.array(trial['variances']) for trial in trials]
    mean_var = np.mean(all_variances, axis=0)

    ax.loglog(positions, mean_var, 'o', markersize=8, color='black', label='Observed')

    # OU fit
    ou_params = data['section4']['model_fits']['ou']
    theta = ou_params['theta']
    sigma2_inf = ou_params['sigma_inf_sq']
    t_smooth = np.logspace(np.log10(min(positions)), np.log10(max(positions)), 100)
    ou_pred = sigma2_inf * (1 - np.exp(-2 * theta * t_smooth))
    ax.loglog(t_smooth, ou_pred, '-', linewidth=2, color='red', label='OU Model')

    # Brownian (linear)
    slope = mean_var[-1] / positions[-1]
    brownian_pred = slope * t_smooth
    ax.loglog(t_smooth, brownian_pred, '--', linewidth=2, color='gray', label='Brownian')

    ax.set_xlabel('Token Position (log scale)')
    ax.set_ylabel('Variance (log scale)')
    ax.set_title('Variance Growth (Log-Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.savefig(OUTPUT_DIR / 'variance_growth.png')
    plt.close()
    print("✓ Generated variance_growth.png")

# ============================================================================
# Generate all figures
# ============================================================================
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)

    generate_fig1_model_comparison()
    generate_fig2_alignment_decay()
    generate_fig3_temperature()
    generate_fig4_domains()
    generate_fig5_summary()
    generate_fig6_variance_raw()
    generate_fig7_ou_prediction()
    generate_fig8_loss_scaling()
    generate_fig9_cross_model()
    generate_stress_test_figure()
    generate_entropy_control_figure()
    generate_variance_growth_loglog()

    print("\n✓ All paper figures generated successfully!")
    print(f"\nGenerated 12 figures in {OUTPUT_DIR}/")
