"""Distribute *Python* objective evaluations across processes/workers.

turboswarm already parallelizes the swarm loop in Rust, but when the objective
itself is expensive and written in Python (a heavy model, a simulation), the
bottleneck is the per-particle Python call. These helpers wrap a per-particle
objective ``f(x) -> float`` into a **vectorized** objective that evaluates the
whole swarm in parallel; pass the result to ``minimize(..., vectorized=True)``::

    from turboswarm.integrations import parallel
    obj = parallel.joblib_objective(expensive_f, n_jobs=-1)
    r = pso.minimize(obj, bounds=[(-5, 5)] * 10, vectorized=True)

Requires joblib (``pip install turboswarm[parallel]``) or dask
(``pip install turboswarm[dask]``).
"""

from __future__ import annotations

import numpy as np


def joblib_objective(fun, n_jobs=-1, backend="loky", **parallel_kwargs):
    """Wrap a per-particle objective into a joblib-parallel vectorized objective.

    Args:
        fun: ``f(x) -> float`` evaluated once per particle (``x`` is a 1-D array).
        n_jobs: passed to ``joblib.Parallel`` (``-1`` = all cores).
        backend: joblib backend (``"loky"``, ``"threading"``, ``"multiprocessing"``).
        **parallel_kwargs: forwarded to ``joblib.Parallel``.

    Returns:
        A vectorized objective ``g(X) -> np.ndarray`` for ``vectorized=True``.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError as exc:  # pragma: no cover - exercised without joblib
        raise ImportError(
            "joblib is required for joblib_objective; "
            "install it with: pip install turboswarm[parallel]"
        ) from exc

    def vectorized(X):
        X = np.asarray(X, dtype=float)
        scores = Parallel(n_jobs=n_jobs, backend=backend, **parallel_kwargs)(
            delayed(fun)(row) for row in X
        )
        return np.asarray(scores, dtype=float)

    return vectorized


def dask_objective(fun, client=None):
    """Wrap a per-particle objective into a Dask-parallel vectorized objective.

    Args:
        fun: ``f(x) -> float`` evaluated once per particle.
        client: an optional ``dask.distributed.Client``; if given, work is
            submitted to that cluster, otherwise the local threaded scheduler is
            used via ``dask.delayed``.

    Returns:
        A vectorized objective ``g(X) -> np.ndarray`` for ``vectorized=True``.
    """
    try:
        import dask
    except ImportError as exc:  # pragma: no cover - exercised without dask
        raise ImportError(
            "dask is required for dask_objective; "
            "install it with: pip install turboswarm[dask]"
        ) from exc

    def vectorized(X):
        X = np.asarray(X, dtype=float)
        if client is not None:
            futures = client.map(fun, list(X))
            scores = client.gather(futures)
        else:
            tasks = [dask.delayed(fun)(row) for row in X]
            scores = dask.compute(*tasks)
        return np.asarray(scores, dtype=float)

    return vectorized
