"""Concrete `StateStore` Protocol implementations.

v0.1 ships exactly one Store: `InMemoryStore`. Production-scale
backends (`DuckDBStore`, `RedisStore`, `HybridStateStore`) are
deferred and tracked in the Decision Queue.

Every Store satisfies the `StateStore` Protocol declared in
`conformal_margin.ontology.ports.state_store`. The Protocol is the
swap mechanism: customers configure which Store the FleetMonitor
uses without touching operator or handler code.
"""
