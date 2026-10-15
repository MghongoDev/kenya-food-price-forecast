# ADR 0007 - Splice mechanics: per-kg normalization and the market map

## Status

Accepted. Amends ADR 0006.

## Context

Building the panel ran into two facts that ADR 0006 left to data inspection.

First, units: the old era's per-kg wholesale series is plain `Maize` (KG, 2006-2022-04, five urban markets), while the successor era's wholesale white maize exists only as `Maize (white, dry)` in 90 KG bags (2021-01 to 2026-08). The old `Maize (white)` wholesale series is bags only, and the successor era has no KG rows. A bag price divided by the bag mass is an ordinary per-kg price, but it is an assumption about every row,and it deserves a record.

Second, markets: the successor era renamed markets and gave them new ids. Old Nairobi (id 184) became Kangemi (Nairobi) (id 4626); Kisumu (186) became Kibuye (Kisumu) (5671); Nakuru (1850) became Wakulima (Nakuru) (5669); Mombasa (191) has several successor retail markets, with Kongowea (Mombasa) (5672) the equivalent of the central wholesale market. Eldoret town (185) kept its id and name. A mapping keyed on ids, not names, ties the two eras of one market together.

## Decision

For the target series, both legs are normalized to KES per kg before the splice: KG rows pass through unchanged, 90 KG rows are divided by 90, any other unit row is dropped. The five market ids map explicitly: 184 to (4626, Kangemi (Nairobi)), 186 to (5671, Kibuye (Kisumu)), 1850 to (5669, Wakulima (Nakuru)), 191 to (5672, Kongowea (Mombasa)); 185 maps to itself. The panel keeps a source_name column recording which naming era each observed row came from, so the splice inspection and later plots can color the switch.

## Consequences

The spliced series is one unit (KES per kg) end to end, and the price-level check at the splice point becomes meaningful: a jump at the switch is now a data intuition to inspect rather than a unit mix-up. The bag-division assumption is stated wherever the series is described. The market map is hand-written and small; if a fifth market joins (or a mapped successor turns out wrong), the map is edited and this ADR is superseded. Mombasa's successor must earn its place in the Phase 1 splice inspection like every other market.
