# Uncertain values

An **uncertain value** \(x^{\pm} = [\,\underline{x},\, \overline{x}\,]\)
represents a quantity known only to lie within an interval — partial
information, somewhere between a crisp value and total ignorance. `turboswarm`
can optimize directly over these variables: the swarm searches the space of
intervals, looking for the vector that minimizes your objective.

Internally each variable is encoded as a **center + spread** pair
\(x^{\pm} = c \pm r\), so the swarm coordinates stay decoupled. The decoded
interval is always kept inside its per-variable `(lower, upper)` limits by a
coupled projection, with an optional extra cap on the half-width (`max_spread`).

## From Python

Use `minimize_grey`. Each variable is an uncertain value \(x^{\pm}\) constrained
to lie within its `bounds`; the swarm searches over its center and spread.

```python
import turboswarm as pso

# Find uncertain values whose midpoints minimize a sphere while staying crisp:
# f rewards both accuracy (centers at 0) and certainty (small spread).
def f(greys):
    centers = [(lo + hi) / 2 for (lo, hi) in greys]
    spreads = [(hi - lo) / 2 for (lo, hi) in greys]
    return sum(c * c for c in centers) + sum(spreads)

r = pso.minimize_grey(f, bounds=(-5, 5), dim=2, seed=42)

print(r.best_position)   # [(lo, hi), (lo, hi)]  intervals, near [(0, 0), (0, 0)]
print(r.best_centers)    # center of each variable, (lo + hi) / 2
print(r.best_spreads)    # half-width of each variable, (hi - lo) / 2
print(r.best_value)      # ~0.0
```

The objective receives the candidate as a `list[tuple[float, float]]` (one pair
per variable) and returns a single `float` — the *whitenized* scalar to
minimize. How you collapse the intervals to that scalar (interval arithmetic, a
whitenization rule, an expected value plus an uncertainty penalty…) is entirely
up to your objective.

### Parameters specific to uncertain-value optimization

- **`bounds`** — the `(lower, upper)` *limits* each interval must
  stay within. Either a list of pairs (one per variable) or a single pair with
  `dim`. The whole decoded interval is kept inside these limits.
- **`max_spread`** — optional extra cap on the half-width of each variable:
  `None` (default, limited only by `bounds`), a single float (broadcast to all
  variables) or a list of floats (one per variable).
- **`representation`** — how each uncertain value is passed to *and* read from the
  objective: `"interval"` (default) gives `(lower, upper)` pairs;
  `"center_spread"` gives `(center, spread)` pairs. It does not affect native
  benchmarks, and the result always exposes both forms.

```python
# Same problem expressed in the center/spread representation, with a hard
# cap on how uncertain each variable may become.
def f(greys):
    return sum(c * c for (c, s) in greys) + sum(s for (c, s) in greys)

r = pso.minimize_grey(
    f,
    bounds=(-5, 5),
    dim=2,
    representation="center_spread",
    max_spread=2.0,
    seed=42,
)
```

All the run-control and PSO arguments of `minimize` (`n_particles`, `max_iter`,
`w`, `c1`, `c2`, `velocity`, `topology`, `seed`, `patience`, `tol`, `max_evals`,
`target`, `max_time`, `v_max`, `record_history`) apply unchanged. Interval bounds
are enforced by projection onto the feasible region, so `bounds_handling` does
not apply.

### Native benchmark

`minimize_grey` also accepts the name of a native grey benchmark, which runs
without the GIL. Currently `"grey_sphere"` (expected sphere plus a unit
uncertainty penalty, optimum `f = 0` at the crisp origin):

```python
r = pso.minimize_grey("grey_sphere", bounds=(-5.12, 5.12), dim=3, seed=1)

# Recommended (center_bound, max_spread, optimum) for the benchmark:
print(pso.grey_benchmark_info("grey_sphere"))
```

The same `grey_sphere` is also available in pure Python in
`turboswarm.benchmarks`.

## From Rust

The space is `GreySpace`; each `Grey` carries center/spread and exposes
`lower()`, `upper()`, whitenization `whiten(λ)` (= `lower + λ·(upper − lower)`)
and interval arithmetic (`+`, `−`, `×`, scaling by a scalar).

```rust
use turboswarm_core::prelude::*;

// Two uncertain variables, each interval limited to [-5, 5], half-width ≤ 5.
let space = GreySpace::new(vec![(-5.0, 5.0); 2], vec![5.0; 2]);

// The objective sees decoded `Grey` values:
let objective = |greys: &[Grey]| -> f64 {
    greys.iter().map(|g| g.center() * g.center() + g.spread()).sum()
};
```

`GreySpace` plugs into the same `Pso` loop as every other search space — the
grey-vs-crisp difference lives entirely in how the space decodes and projects,
not in the optimizer.
