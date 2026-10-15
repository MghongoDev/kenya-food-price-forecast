# ADR 0003 - Hand-rolled split conformal, not MAPIE

## Status

Accepted

## Context

Prediction intervals need calibrating. Split conformal is short enough to write by hand: take residuals from a calibration window of recent origins, take the absolute-error quantile, add and subtract it from the point forecast on the log scale, and optionally scale by per-market volatility. The MAPIE library provides the same idea plus more machinery. Time series break conformal's exchangeability assumption, so any implementation here is an approximation whose limits we report.

## Decision

Write the split conformal logic by hand, one implementation shared by every model, calibrated per horizon. MAPIE stays named as the alternative if the need outgrows the hand-rolled version.

## Consequences

Full control, easy to test, no extra dependency. The exchangeability caveat and any per-market scaling choice must be documented in the coverage report.
