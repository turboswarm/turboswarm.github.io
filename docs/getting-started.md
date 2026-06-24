# Getting started

This tutorial walks from a one-liner to comparing algorithms. It assumes you
have run `maturin develop` (see [Installation](installation.md)).

## 1. Your first optimization

```python
import turboswarm as pso

# Minimize the Rastrigin benchmark in 2D. The function runs natively in Rust.
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=42)

print(r.best_value)        # ~0.0
print(r.best_position)     # ~[0.0, 0.0]
```

`bounds` defines the search box. Two equivalent ways:

```python
# One (min, max) per dimension — each dimension can have its own range:
pso.minimize(f, bounds=[(-5, 5), (0, 100)])      # 2-D, different ranges

# A single (min, max) for every dimension — pass dim:
pso.minimize(f, bounds=(-5.12, 5.12), dim=10)    # 10-D, same range
```

The `seed` makes the run reproducible.

## 2. Your own objective

Pass any Python callable `f(list[float]) -> float`:

```python
# Shifted sphere: optimum at (2, 2, 2), value 0.
r = pso.minimize(lambda x: sum((xi - 2) ** 2 for xi in x),
                 bounds=[(-10, 10)] * 3, seed=1)
```

A native benchmark runs without the GIL and is faster; a Python callable is
re-entered through the GIL on every evaluation but is fully flexible.

### Vectorized objectives

With `vectorized=True`, the objective receives the **whole swarm** per call (a
list of rows, `n_particles x dim`) and returns one value per row. This reduces
the Python round-trips from one-per-particle to one-per-iteration:

```python
import numpy as np
r = pso.minimize(lambda X: np.sum(np.asarray(X) ** 2, axis=1),
                 bounds=[(-5, 5)] * 10, vectorized=True, seed=0)
```

The swarm is handed to your objective as a **NumPy array** (`n_particles x dim`),
built from a contiguous buffer — no per-element Python objects.

!!! note "Honest expectations"
    For an **expensive, vectorizable** objective this matches a fully
    NumPy-based library (measured on par with `pyswarms`). For **cheap**
    objectives the per-iteration framework overhead dominates and an all-NumPy
    library can still be ~1.5–2× faster, since it also vectorizes the swarm
    bookkeeping. See [Comparison](comparison.md).

## 3. Choosing a variant and a topology

```python
r = pso.minimize("ackley", bounds=(-32.768, 32.768), dim=2,
                 velocity="fips", topology="ring", seed=1)
```

- `velocity`: `"inertia"` (default), `"constriction"`, `"fips"` — see [Variants](guide/variants.md).
- `topology`: `"global"` (default), `"ring"`, `"vonneumann"` — see [Topologies](guide/topologies.md).

## 4. Reading the result

`minimize` returns a [`PsoResult`](api/python.md):

| Attribute | Meaning |
|-----------|---------|
| `best_position` | best point found (integers if `integer=True`) |
| `best_value` | objective value at `best_position` |
| `convergence` | best value after each iteration (the convergence curve) |
| `history` | `history[iter][particle][dim]`, for animation (empty if `record_history=False`) |

## 5. Comparing algorithms

```python
runs = {
    "inertia/global": pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                                   velocity="inertia", topology="global", seed=7),
    "fips/ring": pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                              velocity="fips", topology="ring", seed=7),
}
for name, r in runs.items():
    print(name, r.best_value)
```

To plot the comparison, see [Visualization](guide/visualization.md). For a
full feature tour, run `python examples/tour.py` (visualization is optional,
behind `--plot` / `--animate`).

## 6. Run control: early stopping and velocity clamp

Stop early once the swarm stops improving, instead of always running every
iteration:

```python
# Stop when the best value does not improve by more than tol for 20
# consecutive iterations.
r = pso.minimize("sphere", bounds=[(-5.12, 5.12)] * 2,
                 max_iter=1000, patience=20, tol=1e-12, seed=42)
print(len(r.convergence))   # usually far fewer than 1000
```

Clamp the per-component velocity (a classic way to curb overshooting):

```python
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, v_max=1.0, seed=1)
```

Other stop conditions: a **target** value, an **evaluation budget**, or a
**wall-clock budget**:

```python
r = pso.minimize("sphere", bounds=[(-5.12, 5.12)] * 2, max_iter=10000,
                 target=1e-6,       # stop once best_value <= 1e-6
                 max_evals=50_000,  # ...or after 50k objective evaluations
                 max_time=2.0,      # ...or after 2 seconds
                 seed=42)
print(r.stop_reason)   # "target" | "max_evaluations" | "max_time" | ...
print(r.evaluations)   # objective evaluations performed
```

The result reports `stop_reason` and `evaluations`. All of these default to
off, so the standard behavior is unchanged unless you opt in.

A **callback** runs once per iteration — handy for live logging or custom
stopping. It receives `(iteration, best_value)`; return `False` to stop early:

```python
def cb(iteration, best_value):
    print(iteration, best_value)
    return best_value > 1e-8   # keep going until good enough

r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, callback=cb, seed=1)
# r.stop_reason == "callback" if the callback returned False
```
