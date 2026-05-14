"""Ports layer: Protocols that define cross-boundary contracts.

Every contract lives in its own file. Customer adapters implement
these Protocols; library internals depend only on them, never on
concrete adapter classes.

Two kinds of Protocols live here:

- **Data shape Protocols** (`AgentStateLike`, `FleetSnapshotLike`,
  `OperatorSignalLike`): describe the field shape of values that
  flow through the library. No `role` attribute - they are data,
  not components.

- **Component Protocols** (`Operator`, `StateStore`, `AlertHandler`,
  `Filter`, `FeedbackPort`): describe the behaviour of pluggable
  components. Each pins exactly one Role via a `role` class attribute.
"""
