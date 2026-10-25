# <slug>: <one-line title>

- Date: YYYY-MM-DD
- Run group / script: `<group_id>` or `<script + args>`
- Commit: `<sha>` (dirty: no)
- Status: draft | final

## Question
One question. One variable changes per experiment.

## Hypothesis
Expected outcome and why. Written BEFORE looking at results.

## Setup
Provider, $/h, GPU (model, form factor, power limit, driver), software versions,
model @ revision, precision, engine params, workload. Link to the config file.

## Results
Table with the numbers. Units in headers. Sample size per row.
In prose write percentiles as p50@TPOT = 11.2 ms (see CLAUDE.md).

## Sanity check
Compare with a first-principles bound (bandwidth, FLOPs, memory). Explain the gap.

## Observations
What the numbers show, no interpretation.

## Interpretation
Why. Which resource is the bottleneck.

## What did not work / surprises

## Context: reference points
Place the number next to plan section 7 reference points.

## Next
The single next experiment this result suggests.
