# ADR 0002 - CHIRPS rainfall stays in the MVP

## Status

Accepted

## Context

The guide includes rainfall features: 3- and 6-month cumulative totals, plus a calendar-month anomaly against a fixed climatology from training years only. CHIRPS brings an external download, raster sampling around each market, and an ablation run. It is the costliest external piece of the build, and it produces a candidate headline finding: rainfall may help at h=3 and not at h=1.

## Decision

Keep rainfall in Phase 3 as written. Defer only on a structural blocker, such as the download route failing. Rainfall data enters features only up to the forecast origin; no rainfall forecast is used or pretended.

## Consequences

If rainfall adds nothing, "it didn't help" is a recorded, citable result. Schedule risk sits in data access. Testing access happens at the start of Phase 3, not at the end.
