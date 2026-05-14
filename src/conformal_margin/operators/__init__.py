"""Concrete operators and their sub-components.

The library ships one operator in v0.1, `BaselineOperator`, built
on three concrete sub-components:

- `RobustScorer` - median plus scaled Median Absolute Deviation
  (MAD) nonconformity scoring. Robust to outliers in the
  calibration set.
- `ConformalCalibrator` - inductive split conformal p-value with
  exact rank statistic. Preserves the formal
  `P(false positive rate > alpha) <= alpha` coverage guarantee.
- `DriftDetector` - Page-Hinkley streaming test for shifts in the
  mean of the nonconformity-score distribution. Catches when the
  calibration set is no longer exchangeable with current data
  (the precondition for the coverage guarantee).

Sub-operators are concrete classes, not Protocols, in v0.1. A
customer who wants different mathematics implements the
`Operator` Protocol from scratch with their own internal
composition. Sub-operator Protocol extraction is a v0.2+ option
if customer demand surfaces.
"""
