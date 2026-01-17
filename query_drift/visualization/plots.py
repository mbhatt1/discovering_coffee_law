"""Plotting utilities for query drift validation results."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

# Try to use seaborn style if available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Color palette for consistent styling
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "fit_line": "#1B4332",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}

# Font settings
FONT_SETTINGS = {
    "title": {"fontsize": 14, "fontweight": "bold"},
    "label": {"fontsize": 11},
    "tick": {"fontsize": 9},
    "legend": {"fontsize": 9},
    "annotation": {"fontsize": 9},
}


def setup_style() -> None:
    """Set matplotlib style for publication-quality plots.

    Uses seaborn style if available, otherwise falls back to
    a clean matplotlib configuration.
    """
    if HAS_SEABORN:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")

    # Override with custom settings
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["grid"],
        "font.family": "sans-serif",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def add_power_law_fit_annotation(
    ax: plt.Axes,
    exponent: Optional[float],
    r_squared: Optional[float],
    position: Tuple[float, float] = (0.05, 0.95)
) -> None:
    """Add power law fit annotation to a plot axis.

    Args:
        ax: Matplotlib axes object to annotate.
        exponent: The power law exponent (alpha value). Can be None.
        r_squared: The R-squared value of the fit. Can be None.
        position: Tuple of (x, y) coordinates in axes fraction (0-1).
    """
    if exponent is None and r_squared is None:
        return  # Nothing to annotate

    parts = []
    if exponent is not None:
        parts.append(f"$\\alpha$ = {exponent:.3f}")
    else:
        parts.append("$\\alpha$ = N/A")

    if r_squared is not None:
        parts.append(f"$R^2$ = {r_squared:.4f}")
    else:
        parts.append("$R^2$ = N/A")

    annotation_text = "\n".join(parts)

    ax.annotate(
        annotation_text,
        xy=position,
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=FONT_SETTINGS["annotation"]["fontsize"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=COLORS["grid"],
            alpha=0.9,
        ),
    )


def create_comparison_plot(results: Dict[str, Any]) -> plt.Figure:
    """Create a comparison plot showing all exponents in one figure.

    Args:
        results: Dictionary containing experiment results with keys:
            - 'experiments': List of experiment data dictionaries
            - Each experiment should have 'name', 'exponent', 'r_squared',
              and optionally 'expected_exponent'

    Returns:
        matplotlib Figure object with the comparison plot.
    """
    setup_style()

    experiments = results.get("experiments", [])
    if not experiments:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No experiment data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [exp.get("name", f"Exp {i}") for i, exp in enumerate(experiments)]
    measured = [exp.get("exponent", 0) for exp in experiments]
    expected = [exp.get("expected_exponent", exp.get("exponent", 0)) for exp in experiments]
    r_squared = [exp.get("r_squared", 0) for exp in experiments]

    x = np.arange(len(names))
    width = 0.35

    # Bar plots for measured vs expected
    bars_measured = ax.bar(
        x - width/2, measured, width,
        label="Measured $\\alpha$",
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=1,
    )
    bars_expected = ax.bar(
        x + width/2, expected, width,
        label="Expected $\\alpha$",
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=1,
        alpha=0.7,
    )

    # Add R-squared values as text above bars
    for i, (bar, r2) in enumerate(zip(bars_measured, r_squared)):
        height = bar.get_height()
        ax.annotate(
            f"$R^2$={r2:.3f}",
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
        )

    ax.set_xlabel("Experiment", **FONT_SETTINGS["label"])
    ax.set_ylabel("Power Law Exponent ($\\alpha$)", **FONT_SETTINGS["label"])
    ax.set_title("Comparison of Power Law Exponents Across Experiments", **FONT_SETTINGS["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(loc="upper right", **FONT_SETTINGS["legend"])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


class ResultVisualizer:
    """Visualizer for query drift validation experiment results.

    Creates publication-quality plots for power law validation experiments,
    including 2x2 subplot grids and summary figures.

    Attributes:
        output_dir: Directory where plots will be saved.
        save_plots: Whether to save plots to disk.
        show_plots: Whether to display plots interactively.
        dpi: Resolution for saved figures.
    """

    def __init__(
        self,
        output_dir: str,
        save_plots: bool = True,
        show_plots: bool = True,
        dpi: int = 150
    ) -> None:
        """Initialize the ResultVisualizer.

        Args:
            output_dir: Directory path where plots will be saved.
            save_plots: If True, save plots to output_dir.
            show_plots: If True, display plots using plt.show().
            dpi: Resolution (dots per inch) for saved figures.
        """
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.dpi = dpi

        # Create output directory if it doesn't exist
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store the current figure for save_summary_figure
        self._current_figure: Optional[plt.Figure] = None

        # Apply consistent styling
        setup_style()

    def plot_all_experiments(
        self,
        experiments: List[Dict[str, Any]],
        title: str = "Query Drift Validation"
    ) -> plt.Figure:
        """Plot all experiments in a 2x2 subplot grid.

        Args:
            experiments: List of experiment dictionaries, each containing:
                - 'name': Experiment name/label
                - 'x_data': X-axis data (e.g., frequencies, ranks)
                - 'y_data': Y-axis data (e.g., counts, probabilities)
                - 'exponent': Fitted power law exponent
                - 'r_squared': R-squared value of fit
                - 'fit_x': X values for fit line (optional)
                - 'fit_y': Y values for fit line (optional)
                - 'x_label': Label for x-axis (optional)
                - 'y_label': Label for y-axis (optional)
            title: Main title for the figure.

        Returns:
            matplotlib Figure object with 2x2 subplot grid.
        """
        setup_style()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Color cycle for different experiments
        color_cycle = [COLORS["primary"], COLORS["secondary"],
                       COLORS["tertiary"], COLORS["quaternary"]]

        for i, ax in enumerate(axes):
            if i < len(experiments):
                exp = experiments[i]
                color = color_cycle[i % len(color_cycle)]
                self._plot_single_experiment(ax, exp, color)
            else:
                # Empty subplot - hide it
                ax.set_visible(False)

        # Add main title
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        plt.tight_layout()

        # Store for potential saving
        self._current_figure = fig

        # Save if configured
        if self.save_plots:
            filename = title.lower().replace(" ", "_") + ".png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")

        # Show if configured
        if self.show_plots:
            plt.show()

        return fig

    def _plot_single_experiment(
        self,
        ax: plt.Axes,
        experiment: Dict[str, Any],
        color: str
    ) -> None:
        """Plot a single experiment on the given axes.

        Args:
            ax: Matplotlib axes to plot on.
            experiment: Experiment data dictionary.
            color: Color for the data points.
        """
        name = experiment.get("name", "Experiment")
        x_data = experiment.get("x_data", [])
        y_data = experiment.get("y_data", [])
        exponent = experiment.get("exponent")  # Keep as None if not present
        r_squared = experiment.get("r_squared")  # Keep as None if not present
        x_label = experiment.get("x_label", "X")
        y_label = experiment.get("y_label", "Y")

        # Convert to numpy arrays
        x_data = np.array(x_data) if not isinstance(x_data, np.ndarray) else x_data
        y_data = np.array(y_data) if not isinstance(y_data, np.ndarray) else y_data

        # Plot data points (log-log scale)
        if len(x_data) > 0 and len(y_data) > 0:
            ax.scatter(
                x_data, y_data,
                c=color,
                alpha=0.6,
                s=30,
                label="Data",
                edgecolors="white",
                linewidth=0.5,
            )

            # Plot fit line if provided
            fit_x = experiment.get("fit_x")
            fit_y = experiment.get("fit_y")

            if fit_x is not None and fit_y is not None and exponent is not None:
                ax.plot(
                    fit_x, fit_y,
                    color=COLORS["fit_line"],
                    linewidth=2,
                    linestyle="--",
                    label=f"Fit ($\\alpha$={exponent:.3f})",
                )
            elif len(x_data) > 1 and exponent is not None and exponent != 0:
                # Generate fit line from exponent if not provided
                try:
                    x_min, x_max = x_data.min(), x_data.max()
                    if x_min > 0 and x_max > 0 and y_data[0] > 0:
                        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 50)
                        # Use the first data point as reference
                        y_fit = y_data[0] * (x_fit / x_data[0]) ** (-exponent)
                        ax.plot(
                            x_fit, y_fit,
                            color=COLORS["fit_line"],
                            linewidth=2,
                            linestyle="--",
                            label=f"Fit ($\\alpha$={exponent:.3f})",
                        )
                except (ValueError, RuntimeWarning):
                    pass  # Skip fit line if calculation fails

            # Set log scale only if data supports it
            if np.all(x_data > 0):
                ax.set_xscale("log")
            if np.all(y_data > 0):
                ax.set_yscale("log")

        # Labels and title
        ax.set_xlabel(x_label, **FONT_SETTINGS["label"])
        ax.set_ylabel(y_label, **FONT_SETTINGS["label"])
        ax.set_title(name, **FONT_SETTINGS["title"])

        # Add power law annotation
        add_power_law_fit_annotation(ax, exponent, r_squared)

        # Legend and grid
        ax.legend(loc="upper right", **FONT_SETTINGS["legend"])
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=FONT_SETTINGS["tick"]["fontsize"])

    def save_summary_figure(self, filename: str) -> Optional[str]:
        """Save the current figure to a file.

        Args:
            filename: Name of the file (with or without extension).
                     If no extension provided, .png will be added.

        Returns:
            Full path to the saved file, or None if no figure to save.
        """
        if self._current_figure is None:
            print("No figure available to save. Run plot_all_experiments first.")
            return None

        # Ensure filename has extension
        if not filename.endswith((".png", ".pdf", ".svg", ".jpg")):
            filename += ".png"

        filepath = self.output_dir / filename
        self._current_figure.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        print(f"Figure saved to: {filepath}")
        return str(filepath)

    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a text summary report of the validation results.

        Args:
            results: Dictionary containing experiment results with keys:
                - 'experiments': List of experiment data dictionaries
                - 'overall_status': Overall validation status (optional)
                - 'timestamp': When the validation was run (optional)

        Returns:
            Formatted string containing the summary report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("QUERY DRIFT VALIDATION SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Timestamp if available
        timestamp = results.get("timestamp", "Not specified")
        lines.append(f"Timestamp: {timestamp}")
        lines.append("")

        # Overall status
        overall_status = results.get("overall_status", "Unknown")
        lines.append(f"Overall Status: {overall_status}")
        lines.append("-" * 60)
        lines.append("")

        # Individual experiment results
        experiments = results.get("experiments", [])
        lines.append("EXPERIMENT RESULTS:")
        lines.append("-" * 40)

        if not experiments:
            lines.append("No experiments found in results.")
        else:
            for i, exp in enumerate(experiments, 1):
                name = exp.get("name", f"Experiment {i}")
                exponent = exp.get("exponent", "N/A")
                expected = exp.get("expected_exponent", "N/A")
                r_squared = exp.get("r_squared", "N/A")
                status = exp.get("status", "Unknown")

                lines.append(f"\n{i}. {name}")
                lines.append(f"   Measured Exponent:  {exponent:.4f}" if isinstance(exponent, (int, float)) else f"   Measured Exponent:  {exponent}")
                lines.append(f"   Expected Exponent:  {expected:.4f}" if isinstance(expected, (int, float)) else f"   Expected Exponent:  {expected}")
                lines.append(f"   R-squared:          {r_squared:.4f}" if isinstance(r_squared, (int, float)) else f"   R-squared:          {r_squared}")
                lines.append(f"   Status:             {status}")

                # Calculate error if possible
                if isinstance(exponent, (int, float)) and isinstance(expected, (int, float)) and expected != 0:
                    error_pct = abs(exponent - expected) / abs(expected) * 100
                    lines.append(f"   Error:              {error_pct:.2f}%")

        lines.append("")
        lines.append("-" * 60)

        # Summary statistics
        if experiments:
            exponents = [exp.get("exponent") for exp in experiments if isinstance(exp.get("exponent"), (int, float))]
            r_squared_vals = [exp.get("r_squared") for exp in experiments if isinstance(exp.get("r_squared"), (int, float))]

            if exponents:
                lines.append("\nSUMMARY STATISTICS:")
                lines.append(f"   Mean Exponent:      {np.mean(exponents):.4f}")
                lines.append(f"   Std Exponent:       {np.std(exponents):.4f}")

            if r_squared_vals:
                lines.append(f"   Mean R-squared:     {np.mean(r_squared_vals):.4f}")
                lines.append(f"   Min R-squared:      {np.min(r_squared_vals):.4f}")

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)

        # Optionally save the report
        if self.save_plots:
            report_path = self.output_dir / "summary_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {report_path}")

        return report
