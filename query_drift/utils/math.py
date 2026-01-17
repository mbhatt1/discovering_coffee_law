"""
Mathematical utilities for power law fitting and statistical analysis.
"""

import numpy as np
from scipy import optimize
from scipy import stats
from typing import Optional
from dataclasses import dataclass


@dataclass
class PowerLawFit:
    """Results from power law fitting."""
    amplitude: float
    exponent: float
    offset: float
    r_squared: float
    params: np.ndarray
    covariance: Optional[np.ndarray]
    std_errors: Optional[np.ndarray]


def power_law(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:
    """
    Power law function: y = a * x^b + c

    Args:
        x: Input values
        a: Amplitude
        b: Exponent
        c: Offset (default 0)

    Returns:
        Computed power law values
    """
    return a * np.power(x.astype(float), b) + c


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    with_offset: bool = False,
    bounds: Optional[tuple] = None
) -> PowerLawFit:
    """
    Fit power law to data using nonlinear least squares.

    Args:
        x: Independent variable values
        y: Dependent variable values
        with_offset: Whether to fit y = a*x^b + c (True) or y = a*x^b (False)
        bounds: Optional bounds for parameters [(a_min, b_min, c_min), (a_max, b_max, c_max)]

    Returns:
        PowerLawFit with fitted parameters and statistics
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Clean data
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
    if with_offset:
        mask &= True  # Allow any y values with offset
    else:
        mask &= (y > 0)  # Need positive y for log fitting without offset

    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        raise ValueError(f"Not enough valid data points for fitting: {len(x_clean)} < 3")

    # Initial guess from log-linear regression
    try:
        log_x = np.log(x_clean)
        log_y = np.log(np.maximum(y_clean, 1e-10))
        slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
        initial_a = np.exp(intercept)
        initial_b = slope
    except Exception:
        initial_a = 1.0
        initial_b = -0.5

    if with_offset:
        # Fit a * x^b + c
        p0 = [initial_a, initial_b, 0.0]

        if bounds is None:
            bounds = ([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])

        try:
            params, pcov = optimize.curve_fit(
                power_law, x_clean, y_clean,
                p0=p0, bounds=bounds, maxfev=10000
            )
        except Exception:
            # Fallback to initial guess
            params = np.array(p0)
            pcov = None

        amplitude, exponent, offset = params
    else:
        # Fit a * x^b using log-linear regression (more stable)
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)

        slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, log_y)

        params = np.array([np.exp(intercept), slope])
        amplitude = np.exp(intercept)
        exponent = slope
        offset = 0.0
        pcov = None

    # Compute R²
    if with_offset:
        y_pred = power_law(x_clean, *params)
    else:
        y_pred = amplitude * np.power(x_clean, exponent)

    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute standard errors
    if pcov is not None:
        try:
            std_errors = np.sqrt(np.diag(pcov))
        except Exception:
            std_errors = None
    else:
        std_errors = None

    return PowerLawFit(
        amplitude=amplitude,
        exponent=exponent,
        offset=offset,
        r_squared=r_squared,
        params=params,
        covariance=pcov,
        std_errors=std_errors
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def estimate_hurst_exponent(
    variances: np.ndarray,
    positions: np.ndarray
) -> tuple[float, float]:
    """
    Estimate Hurst exponent from variance growth.

    For fractional Brownian motion: σ²(t) ∝ t^(2H)

    Args:
        variances: Variance values at different positions
        positions: Corresponding position values

    Returns:
        (H, r_squared): Hurst exponent and goodness of fit
    """
    fit = fit_power_law(positions, variances)
    hurst = fit.exponent / 2.0  # 2H -> H
    return hurst, fit.r_squared


def bootstrap_confidence_interval(
    x: np.ndarray,
    y: np.ndarray,
    fit_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    with_offset: bool = False
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for power law exponent.

    Args:
        x: Independent variable
        y: Dependent variable
        fit_func: Fitting function (should return PowerLawFit)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        with_offset: Whether to use offset in fitting

    Returns:
        (lower_bound, point_estimate, upper_bound) for the exponent
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    exponents = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]

        try:
            fit = fit_func(x_boot, y_boot, with_offset=with_offset)
            exponents.append(fit.exponent)
        except Exception:
            continue

    if len(exponents) < 10:
        raise ValueError("Too few successful bootstrap fits")

    exponents = np.array(exponents)

    alpha = 1 - confidence
    lower = np.percentile(exponents, 100 * alpha / 2)
    upper = np.percentile(exponents, 100 * (1 - alpha / 2))
    point = np.median(exponents)

    return lower, point, upper


def detrended_fluctuation_analysis(
    series: np.ndarray,
    min_window: int = 4,
    max_window: Optional[int] = None,
    num_windows: int = 20
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Detrended Fluctuation Analysis (DFA) to estimate Hurst exponent.

    Args:
        series: Time series data
        min_window: Minimum window size
        max_window: Maximum window size (default: len(series) // 4)
        num_windows: Number of window sizes to test

    Returns:
        (window_sizes, fluctuations, H): Arrays and estimated Hurst exponent
    """
    n = len(series)
    if max_window is None:
        max_window = n // 4

    # Cumulative sum (integration)
    cumsum = np.cumsum(series - np.mean(series))

    # Window sizes (logarithmically spaced)
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num_windows).astype(int)
    )

    fluctuations = []

    for window in window_sizes:
        # Number of non-overlapping windows
        num_segments = n // window
        if num_segments < 2:
            continue

        rms_values = []

        for i in range(num_segments):
            start = i * window
            end = start + window
            segment = cumsum[start:end]

            # Fit linear trend
            x = np.arange(window)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # RMS of detrended segment
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)

        fluctuations.append(np.mean(rms_values))

    window_sizes = window_sizes[:len(fluctuations)]
    fluctuations = np.array(fluctuations)

    # Fit power law to get Hurst exponent
    fit = fit_power_law(window_sizes, fluctuations)
    hurst = fit.exponent

    return window_sizes, fluctuations, hurst
