"""conformal-margin ontology: canonical vocabulary.

The canonical vocabulary of the conformal-margin library: closed enum
sets, exception hierarchy, and Protocol contracts that customers and
library internals share.

Sources under this submodule import only from the Python standard
library (no msgspec, numpy, scipy, anyio, pydantic, structlog).
Downstream code that only needs vocabulary-level typechecking can
import from `conformal_margin.ontology` without exercising the
heavier runtime modules.
"""

VERSION = "0.1.0.dev0"
