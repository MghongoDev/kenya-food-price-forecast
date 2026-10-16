# The alert that catches three times more spikes than momentum

A spike alert only earns its place if it beats the simplest rule you could write without a model. Ours does — on one axis. It catches 75% of price spikes three months ahead. The momentum rule catches 26%. But the model is less precise: 30% of its alerts are real spikes, against 36% for momentum. Whether that trade is worth making depends on what a false alarm costs against a missed spike. This post explains how we got there and what would change our mind.

## The problem

White maize is the staple grain in Kenya. A price spike — a jump of 15% or more in a single month — hits households and agencies that plan food assistance months ahead. We built a forecasting system that predicts prices one to three months ahead for five markets, and an alert rule that fires when a spike looks likely.

The alert rule is simple: fire when the upper bound of the prediction interval implies a rise above the spike threshold. The interval comes from split conformal prediction, a method that calibrates the width of the interval on past errors so that it covers the true price a fixed share of the time. The rule is conservative by construction — it fires only when the entire plausible range of outcomes sits above the threshold.

## The comparison

The naive rule is the simplest thing you could do without a model: fire when the price already rose more than 10% last month. It needs no training, no features, no intervals. If the model alert cannot beat it, the honest headline is that the model is not worth the complexity.

We scored both rules on the same set of forecast origins and horizons, against the same definition of a spike. The model alert catches 75% of spikes; momentum catches 26%. The model alert is right 30% of the time it fires; momentum is right 36% of the time.

So the model does not dominate. It trades precision for recall. It catches nearly three times as many spikes, but more of its alerts are false alarms.

## Why the trade exists

The model alert fires when the interval upper bound crosses the threshold. The interval is wide — it has to cover 80% of outcomes — so the upper bound sits well above the point forecast. When the point forecast is rising but uncertain, the upper bound can cross the threshold even though the most likely outcome is a small rise. That is where the false alarms come from.

Momentum fires only when the price has already jumped. It is precise because a jump that has already happened is more likely to be followed by another. But it misses most spikes because most spikes do not follow a jump — they follow a gradual rise or a weather shock.

## The cost question

Whether the trade is worth making depends on the cost ratio. If a false alarm costs the same as a missed spike, the cost-optimal threshold is 0.35 — fire conservatively, accept more false alarms to catch more spikes. If a missed spike costs five times a false alarm, the threshold drops to 0.00 — fire aggressively, catch almost every spike, accept that most alerts are false.

We do not know the real costs. They would need to come from a response agency that has measured the cost of wasted effort against the cost of being caught unprepared. Until then, the cost analysis is a framework, not a recommendation.

## What would change our mind

A longer rainfall record could show the rainfall gain at short horizons was noise. A structural break — a new import policy, a currency move — could make the past a poor guide. A market with long observed gaps would starve the features and the calibration. And if a response agency supplied real costs, the cost-optimal threshold would stop being a framework and become a number.

The model alert is not a replacement for judgement. It is a tool that catches more spikes than the simplest rule, at a price. The price is worth paying when missed spikes are expensive. The price is not worth paying when false alarms are. The data cannot tell us which world we are in. That is a decision for the people who respond.
