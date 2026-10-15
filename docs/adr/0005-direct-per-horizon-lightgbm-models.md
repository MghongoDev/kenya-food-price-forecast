# ADR 0005 - Direct per-horizon LightGBM models

## Status

Accepted

## Context

The main model must produce forecasts for h = 1, 2, and 3 and sit in an honest backtest. Two shapes exist: recursive, where the model feeds its own outputs forward, and direct, where each horizon has its own model trained to predict t+h from data at t. Recursion compounds error and complicates the leakage story. Direct models keep the leakage story simple: every feature is computed at the origin.

## Decision

Train a direct model per horizon. One global model across markets, with market identity as a categorical, trained on the log-change target, using lag, rolling and seasonal features plus rainfall at or before the origin.

## Consequences

Three model artifacts and a simple leakage story. Recursion stays possible later for live multi-step runs. Tuning stays light: a small grid, validated on a time-ordered block, never on the test period.
