"""Export optimization results as pandas ``DataFrame``s for analysis and
reporting.

    from turboswarm.integrations import pandas as ts_pandas
    ts_pandas.convergence_dataframe(result)   # one row per iteration
    ts_pandas.history_dataframe(result)       # one row per (iteration, particle)

Requires pandas (``pip install turboswarm[pandas]``).
"""

from __future__ import annotations


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised without pandas
        raise ImportError(
            "pandas is required for turboswarm.integrations.pandas; "
            "install it with: pip install turboswarm[pandas]"
        ) from exc
    return pd


def convergence_dataframe(result):
    """Return the convergence curve as a ``DataFrame`` with columns
    ``iteration`` and ``best_value`` (the global best after each iteration)."""
    pd = _require_pandas()
    conv = result.convergence
    return pd.DataFrame({"iteration": range(len(conv)), "best_value": conv})


def history_dataframe(result):
    """Return the per-particle position history as a tidy ``DataFrame``.

    One row per (iteration, particle), with columns ``iteration``, ``particle``
    and one ``x0..x{d-1}`` column per dimension. Requires the optimization to
    have been run with ``record_history=True`` (the default).
    """
    pd = _require_pandas()
    history = result.history
    if not history:
        raise ValueError(
            "result.history is empty; run minimize(..., record_history=True) "
            "to export the per-particle history."
        )
    records = []
    for iteration, particles in enumerate(history):
        for particle, position in enumerate(particles):
            row = {"iteration": iteration, "particle": particle}
            row.update({f"x{d}": v for d, v in enumerate(position)})
            records.append(row)
    return pd.DataFrame.from_records(records)
