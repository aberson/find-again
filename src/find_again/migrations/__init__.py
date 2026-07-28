"""SQL migration files applied by :mod:`find_again.db`.

Each ``NNNN_name.sql`` file is one migration; the leading integer is its version.
The runner (:func:`find_again.db.apply_migrations`) applies pending migrations in
ascending version order, each as one atomic transaction, and records the highest
applied version in ``PRAGMA user_version``.

Each file's SQL is run verbatim via :meth:`sqlite3.Connection.executescript`, so a
migration may contain any number of statements and SQLite's own parser handles
them -- including ``;`` inside string literals and multi-statement trigger bodies.
There is no hand-rolled statement splitter, so migration authors have no ``;``
restriction to remember.
"""

__all__: list[str] = []
