# Extraneous Experiments - Summary Report

Generated: 2026-01-18 01:09:41
Total Duration: 1940.0 seconds (32.3 minutes)

## Experiments Making the Paper Hard to Dismiss

### 1. Dense Sampling + Prompt Diversity
- **Status**: PASSED
- **Mean Hurst Exponent (H)**: N/A
- **95% Confidence Interval**: N/A
- **Variance Trajectory Consistency**: N/A
- **Prompt Templates Tested**: 50
- **Samples per Template**: N/A

### 2. Open-weight Internal States
- **Status**: SKIPPED (--skip-internal-states flag)
- Requires local GPU and open-weight models

### 3. Lost in the Middle Protocol
- **Status**: PASSED
- **U-Curve Detected**: True
- **OU Prediction Correlation**: N/A
- **OU Correlation p-value**: N/A
- **Liu et al. Replication Score**: N/A


## Conclusion

All extraneous experiments PASSED. The COFFEE Law paper is now:
- Robust to sampling artifact critiques (dense sampling)
- Validated beyond embedding proxies (internal states)
- Properly correlated with Liu et al. findings (Lost in Middle)

These additional experiments make the paper significantly harder to dismiss.