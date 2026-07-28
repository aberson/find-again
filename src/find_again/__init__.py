"""find-again: local-first full-text retrieval over development-memory artifacts.

Step 1 scaffold: typed shapes (:mod:`find_again.models`), explicit-root
configuration and the two-layer secret-exclusion policy
(:mod:`find_again.config`), and a CLI stub (:mod:`find_again.cli`). Storage,
adapters, indexer, and search land in later build steps.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
