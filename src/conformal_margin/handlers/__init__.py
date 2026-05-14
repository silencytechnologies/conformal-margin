"""Concrete `AlertHandler` Protocol implementations.

v0.1 ships `LogHandler` (structlog-backed). Additional handlers
(`FilteredHandler`, `AuditFeedbackHandler`) follow in this folder
once their design questions are resolved.

Every handler satisfies the `AlertHandler` Protocol declared in
`conformal_margin.ontology.ports.alert_handler`. The Protocol is
the swap mechanism: the FleetMonitor dispatches to a tuple of
handlers concurrently inside an `anyio.TaskGroup` and never needs
to know which concrete implementation each handler is.
"""
