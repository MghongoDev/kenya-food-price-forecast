# ADR 0006 - Target series: spliced wholesale white maize in KES per kg

## Status

Accepted. Amended by ADR 0007 (per-kg normalization of bag prices and the explicit old-to-new market-id map).

## Context

Data inspection changed the guide's assumption. WFP renamed its maize series around 2020/21:

- Old name, `Maize (white)`, KG: retail runs 2016 to Dec 2020 in 9 markets; wholesale runs 2010 to Mar 2020 in 5 urban markets.
- New names, `Maize (white, dry)` and `Maize`, KG: retail lives only in Kakuma/Dadaab camp markets from 2021, and `Maize (white, dry)` retail exists only from Dec 2023. Wholesale runs 2021 to Aug 2026 in 12 urban markets.

No single retail series has 8-10 continuous years. The guide's default of retail maize with deep history does not hold. Wholesale, by contrast, splices cleanly per market: Eldoret, Nairobi, Mombasa and Kisumu each have old-name wholesale through early 2020 and new-name wholesale from early 2021 through today. Nairobi carries three overlapping names in 2021-2022, so the splice needs an explicit rule.

Beans shows the same structure: `Beans (dry)` wholesale through Mar 2020, new-name wholesale from 2021 on the same five markets.

## Decision

The MVP target series is wholesale white maize in KES per kg, spliced per market across the naming switch, in the markets that hold both eras (start with Eldoret, Nairobi, Mombasa, Kisumu; Nakuru is judged case by case). The successor name is `Maize (white, dry)`; where two or three names overlap in a market-month, that name wins and the panel build records the choice. The join spans from the last old-name month to the first new-name month as one series. Beans reuses the same rule in the post-MVP step.

Scoring rules stand unchanged: aggregate rows are out, observed months only, never interpolate what scoring touches.

## Consequences

The guide's "retail" wording is replaced by wholesale for the MVP, and this is the recorded reason. The deep history restores the guide's 3-4-year test window, which now fits in 2022-2026. The splice point is a structural risk that Phase 1 must plot and inspect around 2020-2021. The retail camp-market panel is a post-MVP panel, not a mixed series.
