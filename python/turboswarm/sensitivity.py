"""Hyperparameter sensitivity analysis via a Cartesian-product sweep.

Run PSO over every combination of the given hyperparameter value lists and
collect the results, so you can see which hyperparameters actually move the
solution quality. Because PSO is stochastic, each combination can be repeated
over several seeds and aggregated (mean/std/min/max of the best value).

Example::

    import turboswarm as ts
    sweep = ts.sweep(
        "rastrigin", bounds=(-5.12, 5.12), dim=2,
        grid={"w": [0.4, 0.7, 0.9], "c1": [1.0, 2.0]},
        seeds=5, n_particles=30, max_iter=100,
    )
    print(sweep.best())          # combination with the lowest mean best value
    ts.viz.plot_sensitivity(sweep, x="w", y="c1")
"""
import itertools
import logging
import statistics

from .turboswarm_native import minimize

logger = logging.getLogger(__name__)


class SweepResult:
    """Result of a hyperparameter sweep.

    Holds one record per Cartesian-product combination. Each record is a ``dict``
    with the swept hyperparameters plus aggregated statistics of the best
    objective value across the seeds: ``mean``, ``std`` (population), ``min``,
    ``max``, the raw ``values`` list and ``n`` (number of seeds).

    Iterable and indexable over its records; see :meth:`best` and
    :meth:`to_dataframe`.
    """

    def __init__(self, records, param_names):
        self.records = records
        self.param_names = list(param_names)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def best(self, metric="mean"):
        """The record with the lowest ``metric`` (default ``"mean"``)."""
        return min(self.records, key=lambda r: r[metric])

    def to_dataframe(self):
        """Return the records as a pandas ``DataFrame`` (pandas imported lazily).

        Raises:
            ImportError: if pandas is not installed (it is not a hard dependency).
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_dataframe() needs pandas; install it with `pip install pandas`"
            ) from e
        return pd.DataFrame(self.records)

    def __repr__(self):
        return (
            f"SweepResult(combinations={len(self.records)}, "
            f"params={self.param_names})"
        )


def _as_seed_list(seeds):
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError("seeds (int) must be >= 1")
        return list(range(seeds))
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must yield at least one seed")
    return seed_list


def sweep(objective, bounds, grid, *, seeds=1, **kwargs):
    """Sweep PSO over the Cartesian product of hyperparameter values.

    Args:
        objective: objective passed to :func:`turboswarm.minimize` (a callable
            or the name of a native benchmark).
        bounds: bounds passed to ``minimize`` (a list of ``(min, max)`` pairs,
            or a single pair together with ``dim=N`` in ``kwargs``).
        grid (dict[str, list]): maps a hyperparameter name -- any keyword
            accepted by ``minimize``, e.g. ``"w"``, ``"c1"``, ``"c2"``,
            ``"n_particles"``, ``"max_iter"``, ``"velocity"``, ``"topology"`` --
            to the list of values to try. The sweep runs every combination.
        seeds (int | Iterable[int]): repetitions per combination. An ``int``
            means ``range(seeds)``; an iterable is used verbatim. Defaults to
            ``1``; use more for robust, less noisy results.
        **kwargs: fixed parameters forwarded to every ``minimize`` call (e.g.
            ``dim=2, n_particles=30, max_iter=100``). Must not overlap with the
            keys in ``grid``. ``history`` recording is turned off by default
            (only ``best_value`` is needed); pass ``record_history=True`` to
            override.

    Returns:
        SweepResult: one record per combination.
    """
    if not grid:
        raise ValueError("grid must contain at least one hyperparameter")
    overlap = set(grid) & set(kwargs)
    if overlap:
        raise ValueError(f"keys cannot be both swept and fixed: {sorted(overlap)}")
    if "seed" in kwargs:
        raise ValueError("set repetitions with `seeds=`, not a fixed `seed`")

    names = list(grid)
    value_lists = [list(grid[n]) for n in names]
    for n, vals in zip(names, value_lists):
        if not vals:
            raise ValueError(f"grid['{n}'] is empty")
    seed_list = _as_seed_list(seeds)

    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("record_history", False)

    n_combos = 1
    for vals in value_lists:
        n_combos *= len(vals)
    logger.info(
        "sweep: %d combinations x %d seed(s) = %d runs",
        n_combos, len(seed_list), n_combos * len(seed_list),
    )

    records = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(names, combo))
        values = [
            minimize(objective, bounds, seed=s, **params, **call_kwargs).best_value
            for s in seed_list
        ]
        rec = dict(params)
        rec["values"] = values
        rec["n"] = len(values)
        rec["mean"] = statistics.fmean(values)
        rec["std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rec["min"] = min(values)
        rec["max"] = max(values)
        records.append(rec)
    return SweepResult(records, names)
