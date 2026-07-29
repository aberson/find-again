"""find-again: local-first full-text retrieval over development-memory artifacts.

Built so far: typed shapes (:mod:`find_again.models`), explicit-root configuration
and the two-layer secret-exclusion policy (:mod:`find_again.config`), the
SQLite/FTS5 storage layer (:mod:`find_again.db`), the parsing adapters
(:mod:`find_again.adapters`), the incremental indexer + reconciliation
(:mod:`find_again.indexer`), and the ``index``/``status`` CLI verbs
(:mod:`find_again.cli`). Search (``find-again search``) lands in a later build step.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
