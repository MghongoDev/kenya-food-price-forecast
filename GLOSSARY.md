# Glossary

The words this project uses, in plain meaning only. How the code builds each thing lives next to the code.

## Core objects

**Market**
A named trading location in the WFP Kenya price data. Each market has a fixed id and coordinates.

**Month**
The calendar month a price observation belongs to. Prices arrive as market-day dates; the panel groups them by month.

**Price series**
One market, one commodity, one unit, one price type (retail or wholesale), tracked month by month. A single series never mixes two of any of these.

**Panel**
The table that holds every selected price series. One row per market, commodity, and month.

**Observed**
True when the data source recorded an actual price for that series and month. False when the month is missing or the row came from aggregate data. Scoring uses observed rows only.

**Aggregate row**
A price row published with a flag of aggregate rather than actual. Aggregate rows are excluded from scoring.

**Splice**
The join of an old-name series (through March 2020) with its new-name successor (from January 2021) for the same market, unit, and price type. After the join, it counts as one series.

**Publication lag**
The gap between the month a price was observed and the month it became available. The latest actual data currently reaches about one month back.

## Forecasting

**Forecast origin (t)**
The month a forecast is made from. Only data at or before t may feed features.

**Horizon (h)**
How far ahead the forecast predicts, in months. The MVP scores h = 1, 2, and 3.

**Target**
Log price change: the log price at t+h minus the log price at t. Reported back in KES.

**Backtest**
A replay of the forecasting problem on past data. The rolling-origin form steps the origin forward month by month and scores each step on observed months only.

**Baseline**
A simple rule that any model must beat to earn its complexity: naive, seasonal naive, and seasonal mean change.

## Alerts and intervals

**Spike**
A price rise of at least 20 percent in the forecast month compared with the origin month. The threshold can move after checking the base rate, and the final choice is recorded.

**Alert rule**
A fixed rule that turns a forecast into alert or no alert at each origin and horizon.

**False alarm**
An alert fired but no spike followed.

**Missed spike**
A spike happened and no alert fired.

**Data coverage**
The share of months in a window where a series has observed values. Used to select markets.

**Interval coverage**
The share of scored forecasts whose stated 80 percent interval contains the observed price. Watch the name: this word collides with data coverage, and they measure different things.

**Interval half-width**
The distance added on either side of the point forecast on the log scale, set by conformal calibration.
