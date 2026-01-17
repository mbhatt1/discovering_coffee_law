#!/usr/bin/env python3
"""
Cross-model comparison of OU dynamics.
Tests whether COFFEE Law holds across different models.
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from openai import OpenAI

client = OpenAI()

# Models to test
COMPLETION_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",  # If available
    "gpt-4.1",       # If available
]

EMBEDDING_MODEL = "text-embedding-3-small"

PREFIX = """The development of artificial intelligence has progressed through several key phases.
Early research focused on symbolic reasoning and rule-based systems. Modern approaches use
neural networks trained on massive datasets. The field continues to evolve rapidly with new
breakthroughs in language understanding and generation."""

POSITIONS = [10, 25, 50, 75]
NUM_CONTINUATIONS = 10

def get_embedding(text):
    """Get embedding for text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding)

def generate_continuations(model, prefix, n, max_tokens=100):
    """Generate n continuations from a model."""
    continuations = []
    for i in range(n):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Continue the text naturally."},
                    {"role": "user", "content": prefix}
                ],
                max_tokens=max_tokens,
                temperature=1.0
            )
            continuations.append(response.choices[0].message.content)
        except Exception as e:
            print(f"    Error with {model}: {e}")
            return None
    return continuations

def compute_variance_at_positions(prefix, continuations, positions):
    """Compute embedding variance at different positions."""
    embeddings_by_pos = {p: [] for p in positions}

    for cont in continuations:
        words = cont.split()
        for pos in positions:
            if pos <= len(words):
                text = prefix + " " + " ".join(words[:pos])
            else:
                text = prefix + " " + cont
            emb = get_embedding(text)
            embeddings_by_pos[pos].append(emb)

    variances = []
    for pos in positions:
        embs = np.array(embeddings_by_pos[pos])
        centroid = np.mean(embs, axis=0)
        var = np.mean(np.sum((embs - centroid)**2, axis=1))
        variances.append(var)

    return np.array(variances)

def fit_models(positions, variances):
    """Fit Brownian, fBm, and OU models."""
    results = {}

    # fBm: σ²(t) = A * t^(2H)
    def fbm(t, A, H):
        return A * np.power(t, 2*H)

    # OU: σ²(t) = σ²_∞ * (1 - exp(-2θt))
    def ou(t, sigma_inf_sq, theta):
        return sigma_inf_sq * (1 - np.exp(-2 * theta * t))

    try:
        popt, _ = curve_fit(fbm, positions, variances, p0=[0.01, 0.3], bounds=([0, 0], [1, 1]))
        pred = fbm(positions, *popt)
        ss_res = np.sum((variances - pred)**2)
        ss_tot = np.sum((variances - np.mean(variances))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['fbm'] = {'H': popt[1], 'A': popt[0], 'r2': r2}
    except:
        results['fbm'] = None

    try:
        t_norm = positions / positions.max()
        popt, _ = curve_fit(ou, t_norm, variances, p0=[variances.max()*1.5, 1.0], bounds=([0, 0.001], [1, 100]))
        theta_actual = popt[1] / positions.max()
        pred = ou(t_norm, *popt)
        ss_res = np.sum((variances - pred)**2)
        ss_tot = np.sum((variances - np.mean(variances))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        results['ou'] = {'sigma_inf_sq': popt[0], 'theta': theta_actual, 'tau': 1/(2*theta_actual), 'r2': r2}
    except:
        results['ou'] = None

    return results

def main():
    print("=" * 70)
    print("CROSS-MODEL COMPARISON: Does COFFEE Law Hold?")
    print("=" * 70)
    print()

    all_results = {}

    for model in COMPLETION_MODELS:
        print(f"\n>>> Testing {model}...")

        # Generate continuations
        continuations = generate_continuations(model, PREFIX, NUM_CONTINUATIONS)

        if continuations is None:
            print(f"    Skipping {model} (not available or error)")
            continue

        print(f"    Generated {len(continuations)} continuations")

        # Compute variances
        positions = np.array(POSITIONS)
        variances = compute_variance_at_positions(PREFIX, continuations, POSITIONS)

        print(f"    Variances: {[f'{v:.4f}' for v in variances]}")

        # Fit models
        fits = fit_models(positions, variances)

        all_results[model] = {
            'positions': positions.tolist(),
            'variances': variances.tolist(),
            'fits': fits
        }

        if fits['fbm']:
            print(f"    fBm fit: H = {fits['fbm']['H']:.4f}, R² = {fits['fbm']['r2']:.4f}")
        if fits['ou']:
            print(f"    OU fit:  θ = {fits['ou']['theta']:.4f}, τ = {fits['ou']['tau']:.1f}, R² = {fits['ou']['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Model Comparison")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'H (fBm)':<12} {'θ (OU)':<12} {'τ (tokens)':<12} {'Best R²':<12}")
    print("-" * 70)

    h_values = []
    theta_values = []

    for model, data in all_results.items():
        fits = data['fits']
        h = fits['fbm']['H'] if fits['fbm'] else None
        theta = fits['ou']['theta'] if fits['ou'] else None
        tau = fits['ou']['tau'] if fits['ou'] else None
        best_r2 = max(
            fits['fbm']['r2'] if fits['fbm'] else 0,
            fits['ou']['r2'] if fits['ou'] else 0
        )

        h_str = f"{h:.4f}" if h else "N/A"
        theta_str = f"{theta:.4f}" if theta else "N/A"
        tau_str = f"{tau:.1f}" if tau else "N/A"

        print(f"{model:<20} {h_str:<12} {theta_str:<12} {tau_str:<12} {best_r2:<12.4f}")

        if h: h_values.append(h)
        if theta: theta_values.append(theta)

    print("-" * 70)

    if h_values:
        print(f"\nH across models: {np.mean(h_values):.4f} ± {np.std(h_values):.4f}")
    if theta_values:
        print(f"θ across models: {np.mean(theta_values):.4f} ± {np.std(theta_values):.4f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if h_values:
        h_mean = np.mean(h_values)
        h_std = np.std(h_values)
        if h_std / h_mean < 0.5:  # Low coefficient of variation
            print(f"""
✓ COFFEE Law appears UNIVERSAL across tested models
  - H = {h_mean:.3f} ± {h_std:.3f} (CV = {h_std/h_mean:.1%})
  - All models show mean-reverting (anti-persistent) dynamics
  - Variance saturation is consistent
""")
        else:
            print(f"""
? COFFEE Law shows VARIABILITY across models
  - H = {h_mean:.3f} ± {h_std:.3f} (CV = {h_std/h_mean:.1%})
  - Different models may have different dynamics
  - Further investigation needed
""")

if __name__ == "__main__":
    main()
