#!/usr/bin/env python3
"""
Generate publication-quality plots for the COFFEE Law paper.
Uses data from the latest experimental runs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set publication-quality defaults
mpl.rcParams['font.size'] = 11
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6

# Color scheme
COLORS = {
    'primary': '#0072B2',  # Blue
    'secondary': '#D55E00',  # Orange/Red
    'tertiary': '#009E73',  # Green
    'gray': '#666666',
    'light_gray': '#CCCCCC'
}

def load_bulletproof_data():
    """Load data from latest bulletproof experiments."""
    base_path = Path('bulletproof_results_20260117_172001')
    
    with open(base_path / 'stress_test' / 'results.json') as f:
        stress_data = json.load(f)
    
    with open(base_path / 'entropy_drift' / 'results.json') as f:
        entropy_data = json.load(f)
    
    return stress_data, entropy_data

def load_paper_data():
    """Load data from latest paper experiments."""
    base_path = Path('paper_experiments/run_20260117_170920')
    
    with open(base_path / 'master_results.json') as f:
        paper_data = json.load(f)
    
    return paper_data

def plot_stress_test(stress_data, output_path):
    """Create 3-panel stress test visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Extreme-Scale Retrieval Stress Test', fontweight='bold', y=0.995)
    
    data = stress_data['data']
    distractors = np.array(data['distractor_counts'])
    accuracies = np.array(data['mean_accuracies'])
    mrrs = np.array(data['mean_mrr'])
    mean_ranks = np.array(data['mean_ranks'])
    median_ranks = np.array(data['median_ranks'])
    
    # Panel 1: Accuracy
    ax1 = axes[0]
    ax1.semilogx(distractors, accuracies * 100, 'o-', color=COLORS['primary'], 
                 label='Observed', markersize=8, linewidth=2.5)
    ax1.axhline(y=95, color=COLORS['secondary'], linestyle='--', linewidth=2, 
                label='Plateau at 95%')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_ylim([90, 100])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right')
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold', 
             va='top', fontsize=13)
    
    # Panel 2: MRR with OU fit
    ax2 = axes[1]
    ax2.semilogx(distractors, mrrs, 'o', color=COLORS['primary'], 
                 label='Observed MRR', markersize=8)
    
    # OU fit from data
    mrr_fit = data['mrr_fit']
    A = mrr_fit['amplitude']
    k = mrr_fit['decay_rate']
    baseline = mrr_fit['baseline']
    r_squared = mrr_fit['r_squared']
    
    x_fit = np.logspace(np.log10(50), np.log10(100000), 100)
    y_fit = A * np.exp(-k * x_fit) + baseline
    
    ax2.semilogx(x_fit, y_fit, '-', color=COLORS['secondary'], linewidth=2.5,
                 label=f'OU Fit: $A e^{{-kn}}+b$ ($R^2={r_squared:.3f}$)')
    
    ax2.set_ylabel('Mean Reciprocal Rank', fontweight='bold')
    ax2.set_ylim([0.945, 0.96])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower left')
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold', 
             va='top', fontsize=13)
    
    # Panel 3: Mean vs Median Rank
    ax3 = axes[2]
    ax3.loglog(distractors, mean_ranks, 'o-', color=COLORS['primary'], 
               label='Mean Rank', markersize=8, linewidth=2.5)
    ax3.loglog(distractors, median_ranks, 's-', color=COLORS['tertiary'], 
               label='Median Rank', markersize=8, linewidth=2.5)
    ax3.set_xlabel('Number of Distractors', fontweight='bold')
    ax3.set_ylabel('Rank Position', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper left')
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontweight='bold', 
             va='top', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved stress test plot to {output_path}")
    plt.close()

def plot_entropy_control(entropy_data, output_path):
    """Create entropy drift visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Output Entropy: LayerNorm Control', fontweight='bold')
    
    data = entropy_data['data']
    positions = np.array([10, 20, 30, 50, 70, 100, 130, 150])
    mean_entropies = np.array([0.11099749, 0.45130346, 0.84952729, 0.39512512, 
                                0.39211307, 0.64328051, 0.74966488, 0.63653228])
    
    # Plot observed data
    ax.plot(positions, mean_entropies, 'o', color=COLORS['primary'], 
            markersize=10, label='Observed Entropy', zorder=3)
    
    # Saturation fit
    sat_fit = data['saturation_fit']
    x_fit = np.linspace(10, 150, 100)
    A = sat_fit['amplitude']
    rate = sat_fit['rate']
    offset = sat_fit['offset']
    plateau = sat_fit['plateau']
    r2_sat = sat_fit['r_squared']
    
    y_sat = A * (1 - np.exp(-rate * x_fit)) + offset
    ax.plot(x_fit, y_sat, '-', color=COLORS['secondary'], linewidth=2.5,
            label=f'Saturation (OU): $R^2={r2_sat:.2f}$')
    
    # Linear fit
    lin_fit = data['linear_fit']
    slope = lin_fit['slope']
    intercept = lin_fit['intercept']
    r2_lin = lin_fit['r_squared']
    
    y_lin = slope * x_fit + intercept
    ax.plot(x_fit, y_lin, '--', color=COLORS['gray'], linewidth=2,
            label=f'Linear (Brownian): $R^2={r2_lin:.2f}$')
    
    ax.set_xlabel('Context Position (tokens)', fontweight='bold')
    ax.set_ylabel('Shannon Entropy $H(p)$', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add text box with interpretation
    textstr = f'Saturation model: {r2_sat/r2_lin:.1f}× better fit\nControls for LayerNorm artifact'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved entropy control plot to {output_path}")
    plt.close()

def plot_variance_growth(paper_data, output_path):
    """Create variance growth visualization with model comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Variance Growth and Model Comparison', fontweight='bold')
    
    # Left panel: Raw variance data
    variance_data = paper_data['section3']['3.1_variance_growth']
    positions = np.array([10, 20, 30, 50, 75, 100])
    
    for trial_idx, trial in enumerate(variance_data['trials']):
        variances = np.array(trial['variances'])
        ax1.plot(positions, variances, 'o-', alpha=0.6, 
                label=f"Trial {trial_idx}", markersize=6)
    
    # Mean variance
    all_variances = np.array([t['variances'] for t in variance_data['trials']])
    mean_variance = np.mean(all_variances, axis=0)
    ax1.plot(positions, mean_variance, 'ko-', linewidth=2.5, markersize=10,
            label='Mean', zorder=10)
    
    ax1.set_xlabel('Token Position', fontweight='bold')
    ax1.set_ylabel('Embedding Variance $\\sigma^2$', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    ax1.text(0.02, 0.98, '(a) Raw Data', transform=ax1.transAxes, fontweight='bold', 
             va='top', fontsize=12)
    
    # Right panel: Model fits
    model_fits = paper_data['section4']['model_fits']
    
    x_fit = np.linspace(10, 100, 100)
    
    # Brownian
    A_bm = model_fits['brownian']['amplitude']
    y_bm = A_bm * x_fit
    r2_bm = model_fits['brownian']['r_squared']
    ax2.plot(x_fit, y_bm, '--', color=COLORS['gray'], linewidth=2,
            label=f'Brownian: $R^2={r2_bm:.2f}$')
    
    # OU
    sigma_inf_sq = model_fits['ou']['sigma_inf_sq']
    theta = model_fits['ou']['theta']
    tau = model_fits['ou']['relaxation_time']
    r2_ou = model_fits['ou']['r_squared']
    y_ou = sigma_inf_sq * (1 - np.exp(-2 * theta * x_fit))
    ax2.plot(x_fit, y_ou, '-', color=COLORS['secondary'], linewidth=2.5,
            label=f'OU: $R^2={r2_ou:.2f}$, $\\tau={tau:.1f}$ tokens')
    
    # Observed data
    ax2.plot(positions, mean_variance, 'ko', markersize=10, label='Observed', zorder=10)
    
    ax2.set_xlabel('Token Position', fontweight='bold')
    ax2.set_ylabel('Embedding Variance $\\sigma^2$', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    ax2.text(0.02, 0.98, '(b) Model Comparison', transform=ax2.transAxes, 
             fontweight='bold', va='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved variance growth plot to {output_path}")
    plt.close()

def plot_alignment_decay(paper_data, output_path):
    """Create alignment decay visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('Alignment Decay with Task Direction', fontweight='bold')
    
    alignment_data = paper_data['section3']['3.2_alignment_decay']
    
    # Get data from first trial (they're identical)
    trial = alignment_data['trials'][0]
    context_lengths = np.array(trial['context_lengths'])
    alignments = np.array(trial['alignments'])
    beta = trial['fitted_beta']
    r2 = trial['r_squared']
    
    # Log-log plot
    ax.loglog(context_lengths, alignments, 'o', color=COLORS['primary'], 
              markersize=7, label='Observed', alpha=0.7)
    
    # Power law fit
    x_fit = np.logspace(np.log10(context_lengths.min()), 
                        np.log10(context_lengths.max()), 100)
    C0 = alignments[0] * (context_lengths[0] ** beta)
    y_fit = C0 * x_fit ** (-beta)
    ax.loglog(x_fit, y_fit, '-', color=COLORS['secondary'], linewidth=2.5,
              label=f'Fitted: $t^{{-{beta:.3f}}}$ ($R^2={r2:.3f}$)')
    
    # Brownian prediction
    beta_brownian = 0.5
    y_brownian = C0 * x_fit ** (-beta_brownian)
    ax.loglog(x_fit, y_brownian, '--', color=COLORS['gray'], linewidth=2,
              label=f'Brownian: $t^{{-0.5}}$')
    
    ax.set_xlabel('Context Length (characters)', fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add text box
    ratio = 0.5 / beta
    textstr = f'Decay {ratio:.1f}× slower than Brownian prediction'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved alignment decay plot to {output_path}")
    plt.close()

def main():
    """Generate all paper plots."""
    output_dir = Path('paper')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    stress_data, entropy_data = load_bulletproof_data()
    paper_data = load_paper_data()
    
    print("\nGenerating plots...")
    plot_stress_test(stress_data, output_dir / 'fig_stress_test_NEW.png')
    plot_entropy_control(entropy_data, output_dir / 'fig_entropy_control_NEW.png')
    plot_variance_growth(paper_data, output_dir / 'fig_variance_NEW.png')
    plot_alignment_decay(paper_data, output_dir / 'fig_alignment_NEW.png')
    
    print("\n✓ All plots generated successfully!")
    print("\nGenerated files:")
    print("  - fig_stress_test_NEW.png")
    print("  - fig_entropy_control_NEW.png")  
    print("  - fig_variance_NEW.png")
    print("  - fig_alignment_NEW.png")
    
    print("\nKey findings from latest data:")
    print(f"  Stress test: MRR R²={stress_data['data']['mrr_fit']['r_squared']:.4f}")
    print(f"  Entropy: Saturation R²={entropy_data['data']['saturation_fit']['r_squared']:.4f} vs Linear R²={entropy_data['data']['linear_fit']['r_squared']:.4f}")
    print(f"  Variance: Hurst exponent={paper_data['section3']['3.1_variance_growth']['aggregated']['mean_hurst']:.4f}")
    print(f"  Alignment: β={paper_data['section3']['3.2_alignment_decay']['aggregated']['mean_beta']:.4f}")

if __name__ == '__main__':
    main()
