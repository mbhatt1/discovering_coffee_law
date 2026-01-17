"""
Logging utilities for Query Drift experiments.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import time


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """
    Logger specifically designed for experiment tracking.

    Provides structured logging for experiments with timing,
    progress tracking, and result summaries.
    """

    def __init__(
        self,
        name: str,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize experiment logger.

        Args:
            name: Experiment name
            output_dir: Directory for log files
            verbose: Whether to print to console
        """
        self.name = name
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else None

        # Create log file path
        log_file = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(self.output_dir / f"{name}_{timestamp}.log")

        self._logger = get_logger(
            name,
            level=logging.DEBUG if verbose else logging.INFO,
            log_file=log_file
        )

        self._start_times: dict[str, float] = {}
        self._metrics: dict[str, list] = {}

    def info(self, message: str) -> None:
        """Log info message."""
        if self.verbose:
            self._logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    def section(self, title: str) -> None:
        """Log a section header."""
        separator = "=" * 60
        self.info("")
        self.info(separator)
        self.info(title)
        self.info(separator)

    def subsection(self, title: str) -> None:
        """Log a subsection header."""
        self.info(f"\n--- {title} ---")

    def progress(self, current: int, total: int, message: str = "") -> None:
        """Log progress update."""
        percent = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_length - filled)

        msg = f"[{bar}] {percent:5.1f}% ({current}/{total})"
        if message:
            msg += f" - {message}"

        self.info(msg)

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._start_times[name] = time.time()
        self.debug(f"Timer started: {name}")

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name not in self._start_times:
            self.warning(f"Timer '{name}' was not started")
            return 0.0

        elapsed = time.time() - self._start_times[name]
        del self._start_times[name]
        self.debug(f"Timer stopped: {name} ({elapsed:.2f}s)")
        return elapsed

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing a block of code."""
        self.start_timer(name)
        try:
            yield
        finally:
            elapsed = self.stop_timer(name)
            self.info(f"{name}: {elapsed:.2f}s")

    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
        self.debug(f"Metric {name}: {value}")

    def log_result(self, name: str, value, unit: str = "") -> None:
        """Log an experiment result."""
        unit_str = f" {unit}" if unit else ""
        if isinstance(value, float):
            self.info(f"  {name}: {value:.4f}{unit_str}")
        else:
            self.info(f"  {name}: {value}{unit_str}")

    def summary(self) -> None:
        """Print summary of logged metrics."""
        if not self._metrics:
            return

        self.subsection("Metrics Summary")
        for name, values in self._metrics.items():
            import numpy as np
            arr = np.array(values)
            self.info(f"  {name}:")
            self.info(f"    mean: {np.mean(arr):.4f}")
            self.info(f"    std:  {np.std(arr):.4f}")
            self.info(f"    min:  {np.min(arr):.4f}")
            self.info(f"    max:  {np.max(arr):.4f}")


class ProgressBar:
    """Simple progress bar for console output."""

    def __init__(self, total: int, description: str = "", bar_length: int = 40):
        self.total = total
        self.description = description
        self.bar_length = bar_length
        self.current = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
        self._display()

    def _display(self) -> None:
        """Display the progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = (self.current / self.total) * 100

        filled = int(self.bar_length * self.current / self.total) if self.total > 0 else self.bar_length
        bar = "█" * filled + "░" * (self.bar_length - filled)

        desc = f"{self.description}: " if self.description else ""
        print(f"\r{desc}|{bar}| {percent:5.1f}% ({self.current}/{self.total})", end="", flush=True)

        if self.current >= self.total:
            print()  # Newline when complete

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.current < self.total:
            print()  # Ensure newline on exit
