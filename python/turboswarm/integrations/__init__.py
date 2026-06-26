"""Optional integrations with popular libraries.

The integrations turn `turboswarm` into a good citizen of the scientific-Python
and ML ecosystem. They compose as a stack:

  - **NumPy** — the substrate: array ``bounds`` and (vectorized) objectives.
  - **SciPy** (`scipy`) — a drop-in ``scipy.optimize.minimize`` replacement, so
    PSO slots into existing optimization code by changing one line.
  - **scikit-learn** (`sklearn`) — ``PSOSearchCV``, a PSO-driven alternative to
    ``GridSearchCV`` / ``RandomizedSearchCV`` for hyperparameter tuning.
  - **Joblib / Dask** (`parallel`) — distribute *expensive Python* objective
    evaluations across cores/workers.
  - **pandas** (`pandas`) — export the optimization history/convergence as
    ``DataFrame``s for analysis and reporting.

Each integration is **optional** and **lazily imported**, so ``import
turboswarm`` never requires any of these packages. Install the extra you use::

    pip install turboswarm[scipy]      # SciPy wrapper
    pip install turboswarm[sklearn]    # PSOSearchCV
    pip install turboswarm[parallel]   # joblib backend
    pip install turboswarm[pandas]     # DataFrame export
    pip install turboswarm[all]        # everything

Submodules ``scipy``, ``pandas`` and ``parallel`` import only NumPy at module
level (their third-party deps are imported inside the functions that need them),
so they are always importable. ``sklearn`` subclasses scikit-learn estimators,
so it is imported lazily on first access and requires scikit-learn to be
installed.
"""

from . import pandas, parallel, scipy  # noqa: F401  (third-party imports are lazy)

__all__ = ["scipy", "sklearn", "pandas", "parallel"]


def __getattr__(name):
    # Lazily expose the sklearn submodule (it imports scikit-learn at module
    # level) without making `import turboswarm` depend on scikit-learn.
    if name == "sklearn":
        from . import sklearn as _sklearn

        return _sklearn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
