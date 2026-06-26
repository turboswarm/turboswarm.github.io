"""SciPy-compatible wrapper: optimize with PSO using the ``scipy.optimize``
surface, so you can swap your optimizer for ``turboswarm`` by changing one line.

    from turboswarm.integrations import scipy as ts_scipy
    res = ts_scipy.minimize(fun, bounds=[(-5, 5)] * 2)
    res.x, res.fun, res.nit, res.nfev, res.success

Requires SciPy (``pip install turboswarm[scipy]``).
"""

from __future__ import annotations

import numpy as np


def _normalize_bounds(bounds):
    """Convert a ``scipy.optimize.Bounds`` to (min, max) pairs; pass anything
    else (list of pairs, NumPy array, single pair) through to turboswarm, which
    already accepts those forms."""
    lb = getattr(bounds, "lb", None)
    ub = getattr(bounds, "ub", None)
    if lb is not None and ub is not None:
        return [(float(lo), float(hi)) for lo, hi in zip(np.atleast_1d(lb),
                                                         np.atleast_1d(ub))]
    return bounds


def minimize(fun, x0=None, args=(), bounds=None, *, max_iter=100, options=None,
             **pso_kwargs):
    """Minimize ``fun`` with Particle Swarm Optimization, returning a SciPy
    ``OptimizeResult``.

    Mirrors :func:`scipy.optimize.minimize` closely enough to be a drop-in for
    global, gradient-free problems:

        res = minimize(fun, bounds=[(-5, 5)] * 3, seed=0)
        res.x        # best decision vector (np.ndarray)
        res.fun      # best objective value
        res.nit      # number of iterations
        res.nfev     # number of function evaluations
        res.success  # always True if the run completed
        res.message  # the PSO stop reason

    Differences from SciPy (PSO is population-based, not local):

    - ``bounds`` is **required** — the swarm is initialized within it.
    - ``x0`` is **optional**: it is only used to infer the dimension when
      ``bounds`` is a single ``(min, max)`` pair. PSO does not start from a
      single point, so ``x0`` is otherwise ignored.

    Args:
        fun: objective ``f(x, *args) -> float`` (``x`` is a NumPy array).
        x0: optional initial guess; used only to infer the dimension.
        args: extra positional arguments passed to ``fun``.
        bounds: a sequence of ``(min, max)`` pairs, a NumPy array of shape
            ``(dim, 2)``, a single ``(min, max)`` pair (with ``x0`` or
            ``dim=`` to set the dimension), or a ``scipy.optimize.Bounds``.
        max_iter: number of PSO iterations (also accepts SciPy's
            ``options={"maxiter": ...}``).
        options: SciPy-style options dict; ``maxiter`` and ``maxfev``/``maxfun``
            are honored (mapped to ``max_iter`` and ``max_evals``).
        **pso_kwargs: forwarded to :func:`turboswarm.minimize` (e.g.
            ``n_particles``, ``velocity``, ``topology``, ``seed``,
            ``constraints``).

    Returns:
        scipy.optimize.OptimizeResult
    """
    try:
        from scipy.optimize import OptimizeResult
    except ImportError as exc:  # pragma: no cover - exercised without scipy
        raise ImportError(
            "scipy is required for turboswarm.integrations.scipy; "
            "install it with: pip install turboswarm[scipy]"
        ) from exc

    import turboswarm as ts

    if bounds is None:
        raise ValueError(
            "bounds is required: PSO is population-based and initializes the "
            "swarm inside the bounds. Pass bounds=[(lo, hi), ...]."
        )
    bounds = _normalize_bounds(bounds)

    # A single (min, max) pair: take the dimension from x0 if not given.
    is_single_pair = np.ndim(bounds) == 1 and len(bounds) == 2
    if is_single_pair and "dim" not in pso_kwargs and x0 is not None:
        pso_kwargs["dim"] = int(np.size(x0))

    # SciPy carries run-control in `options`.
    if options:
        if "maxiter" in options:
            max_iter = int(options["maxiter"])
        max_evals = options.get("maxfev", options.get("maxfun"))
        if max_evals is not None and "max_evals" not in pso_kwargs:
            pso_kwargs["max_evals"] = int(max_evals)

    def objective(x):
        return float(fun(np.asarray(x, dtype=float), *args))

    result = ts.minimize(objective, bounds=bounds, max_iter=max_iter, **pso_kwargs)

    return OptimizeResult(
        x=np.asarray(result.best_position, dtype=float),
        fun=result.best_value,
        nit=len(result.convergence),
        nfev=result.evaluations,
        success=True,
        status=0,
        message=f"stopped: {result.stop_reason}",
    )
