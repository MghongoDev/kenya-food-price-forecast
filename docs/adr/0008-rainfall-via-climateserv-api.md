# ADR 0008 - Rainfall via the ClimateSERV API, not raster downloads

## Status

Accepted

## Context

ADR 0002 kept CHIRPS rainfall in the MVP and named the download route as
the schedule risk. The route materialized: the UCSB CHC file server
throttles each connection to ~90 KB/s, so the 170 monthly GeoTIFFs
needed for 2010-2016 take hours even with 8 parallel workers. ADR 0002
allows deferring "on a structural blocker, such as the download route
failing" — but the data itself is not the blocker, only the file route.

## Decision

Fetch CHIRPS through the ClimateSERV API
(climateserv.servirglobal.net/api), the official CHIRPS channel listed
on the UCSB CHC page. datatype 0 = UCSB CHIRPS daily rainfall,
operationtype 5 = average over the polygon, intervaltype 0 = daily.
Each market is sampled with a 24-gon approximating the same 25 km
radius circle the raster pixel mask selected; the monthly value sums the
polygon's daily averages over the month. Requests are chunked by
calendar year (the server rejects very long spans) and run with 5
workers, one per market.

## Consequences

Validated against the raster route on 2010-2012 before the switch:
monthly values agree to rounding (Kangemi 2012-04: 401.3 mm both
routes). The rainfall.parquet schema (market_id, month, rain_mm) is
unchanged, so features, the model, and the backtest are untouched.
The raster code and the rasterio pin are deleted; the downloaded
GeoTIFFs are removed. New dependency: none (requests was already
pinned). Residual risk: ClimateSERV is a third-party service that can
change or go down; the raster route is no longer in the tree if it
must be restored.
