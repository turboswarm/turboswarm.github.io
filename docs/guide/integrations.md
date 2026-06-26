# Integrations

`turboswarm` lives in the scientific-Python stack and integrates with the
libraries you already use. Each integration is **optional** and **lazily
imported**, so importing `turboswarm` never requires these packages — install
the extra you need:

```bash
pip install turboswarm[scipy]    # SciPy drop-in wrapper
pip install turboswarm[pandas]   # history / convergence DataFrames
pip install turboswarm[all]      # everything
```

The integrations live under `turboswarm.integrations`.

## NumPy

NumPy is supported out of the box — no extra needed (it is a core dependency).
`bounds` accepts a NumPy array of shape `(dim, 2)` (or `(2,)` with `dim`), and
objectives may return NumPy scalars:

```python
import numpy as np
import turboswarm as pso

bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])          # shape (dim, 2)
r = pso.minimize(lambda x: np.sum(np.asarray(x) ** 2), bounds=bounds, seed=0)
```

For expensive objectives, the **vectorized path** evaluates the whole swarm in
one NumPy call — the objective receives an `(n_particles, dim)` array and
returns an `(n_particles,)` array:

```python
r = pso.minimize(lambda X: np.sum(X ** 2, axis=1),     # X is (n_particles, dim)
                 bounds=[(-5, 5)] * 10, vectorized=True, seed=0)
```

## SciPy

A wrapper mirroring `scipy.optimize.minimize` lets you swap your optimizer for
PSO by changing one line. It returns a SciPy `OptimizeResult` with `.x`, `.fun`,
`.nit`, `.nfev`, `.success` and `.message`:

```python
from turboswarm.integrations import scipy as ts_scipy

def rosen(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

res = ts_scipy.minimize(rosen, bounds=[(-5, 5)] * 3, seed=0)
print(res.x, res.fun)        # ≈ [1, 1, 1], close to the Rosenbrock optimum (f = 0)
print(res.nit, res.nfev, res.success, res.message)
```

Because PSO is population-based rather than local, the contract differs from
SciPy in two honest ways:

- **`bounds` is required** — the swarm is initialized inside it.
- **`x0` is optional** — it is only used to infer the dimension when `bounds` is
  a single `(min, max)` pair; PSO does not start from a single point.

SciPy's `options={"maxiter": ...}` (and `maxfev`/`maxfun`) are honored, and any
extra keyword (`n_particles`, `velocity`, `topology`, `seed`, `constraints`, …)
is forwarded to [`turboswarm.minimize`](../api/python.md). It also accepts a
`scipy.optimize.Bounds` object.

## pandas

Export an optimization result as tidy `DataFrame`s for analysis and reporting:

```python
from turboswarm.integrations import pandas as ts_pandas

r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1)

ts_pandas.convergence_dataframe(r)   # columns: iteration, best_value
ts_pandas.history_dataframe(r)       # columns: iteration, particle, x0, x1, ...
```

`history_dataframe` needs the run to have recorded history (the default,
`record_history=True`); it is one row per (iteration, particle). From there the
whole pandas/Matplotlib stack is available for custom analysis.

## More integrations

scikit-learn (`PSOSearchCV`), Optuna (sampler + comparison), Joblib/Dask
(distributed Python objectives) and a PyTorch example are on the
[roadmap](https://github.com/turboswarm/turboswarm.github.io/issues?q=is%3Aissue+label%3Aintegration).
