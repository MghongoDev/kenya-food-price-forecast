# ADR 0009 - Alert on the conformal upper bound, not the point forecast

## Status

Accepted

## Context

The alert rule must turn a forecast into alert / no alert. The guide
offers two candidates: fire when the conformal interval's upper bound
at t+h implies a rise above the spike threshold, or when the median
(point forecast) does. The point forecast is the more aggressive rule;
it fires whenever the predicted rise crosses the threshold, which
maximises recall but admits many false alarms when the prediction is
wrong. The upper bound fires only when the *entire plausible range* of
outcomes sits above the threshold.

## Decision

Fire on the conformal upper bound: alert when
upper[t+h] >= (1 + alert_threshold) * price[t]. The alert threshold is
a separate knob from the spike threshold; it is swept to produce a
precision-recall curve, and the operating point is picked from that
curve (ticket #12's cost analysis, not this ADR).

## Consequences

The rule is conservative by construction: it fires less often than the
median rule, trading recall for precision. At the spike threshold the
measured operating point is precision 0.30 / recall 0.75 — most spikes
are caught, but most alerts are false alarms. That trade-off is the
honest headline for Phase 5 and is why the cost analysis (ADR
0009's downstream) exists: the right operating point depends on the
false-alarm-to-miss ratio, which this ADR does not assume.

The naive momentum rule (price[t-1] >= 1.1 * price[t-2], "price
already rose more than 10% last month") is the comparison baseline for
whether the model alert earns its complexity. Measured at the spike
threshold: model precision 0.30 / recall 0.75 vs. naive 0.22 / 0.16.
The model beats momentum on both axes.
