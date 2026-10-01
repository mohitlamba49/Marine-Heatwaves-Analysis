# Publication figure generation — pseudocode

This file documents figure assembly without embedding journal-specific styling code.

## Observed annual characteristics

```text
for representative Lakshadweep and Andaman sites:
    create a 3 x 2 panel
    plot annual event frequency
    plot mean event duration only in event years
    plot mean cumulative climatological-mean anomaly only in event years
    plot mean maximum intensity only in event years
    plot annual MHW days
    leave unused sixth axis blank
    overlay Sen line and annotate tau, p-value, slope, and trend classification
```

## Model comparison panel

```text
main text:
    choose the two prespecified representative sites
    include GAM and stacking ensemble at each site in a 2 x 2 panel
    use identical axes and confidence-band conventions

supplement:
    show remaining model/site observed-versus-predicted plots
    preserve chronological date axes
    state how every confidence band was calculated
```

## Rules

- Do not draw a continuous line across missing event-level annual means.
- Do not label a residual band as complete predictive uncertainty.
- Call every figure in the text as part of a grammatical sentence.
- Keep site coordinates, region names, model names, units, and test dates in captions.

