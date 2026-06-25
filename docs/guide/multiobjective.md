# Multi-objective optimization (MOPSO)

When you have several conflicting objectives there is no single best solution,
but a set of trade-offs — the **Pareto front**. `minimize_multi` returns an
approximation of that front using MOPSO (Coello Coello & Lechuga, 2004): an
external archive of non-dominated solutions, leader selection by crowding
distance, and Pareto-dominance updates of the personal bests.

## Example

Minimize two conflicting objectives (pulling `x` toward 0 and toward 2):

```python
import turboswarm as pso

front = pso.minimize_multi(
    lambda x: [sum(xi**2 for xi in x), sum((xi - 2) ** 2 for xi in x)],
    bounds=[(-5, 5)] * 2,
    n_particles=100, max_iter=100, archive_size=100, seed=42,
)

print(len(front))            # number of non-dominated solutions
front.positions              # list[list[float]] — decision vectors
front.objectives             # list[list[float]] — their objective values
```

The objective returns a **list of values** (all minimized). The decision set
here is `x ∈ [0, 2]`, and the front spans from `(f1≈0, f2≈4·dim)` to
`(f1≈4·dim, f2≈0)`.

## Parameters

| Parameter | Meaning |
|-----------|---------|
| `archive_size` | maximum size of the returned front (pruned by crowding distance) |
| `velocity` | `"inertia"` or `"constriction"` — single-leader rules (FIPS does not apply) |
| `mutation_rate` | turbulence strength in `[0, 1]` (default `0.1`); improves front spread, `0` disables |
| `grid_divisions` | archive diversity: `None` (default) = crowding distance; an int `d` = Coello's adaptive grid with `d` cells per objective |
| `integer`, `binary`, `var_types` | same as in `minimize` (mixed problems supported) |

### Archive diversity: crowding vs grid

The external archive is kept diverse in one of two ways. By default it keeps the
most isolated members by **NSGA-II crowding distance**. Passing
`grid_divisions=d` switches to the **adaptive hypercube grid** from the original
MOPSO paper (Coello Coello & Lechuga): objective space is split into `d` cells
per objective (re-fitted to the archive each iteration); pruning removes members
from the most crowded cell and leaders are drawn towards sparser cells. The grid
tends to spread the front more evenly, especially for larger archives.

```python
front = pso.minimize_multi(objectives, bounds=bounds, grid_divisions=30, seed=0)
```

## Visualizing the front

For two objectives:

```python
import matplotlib.pyplot as plt
pso.viz.plot_pareto(front)
plt.show()
```

## Measuring front quality: hypervolume

The **hypervolume** is the volume of objective space dominated by the front and
bounded by a *reference* point (for minimization — larger is better). It rewards
both convergence and spread in a single number, with no reference front needed:

```python
hv = front.hypervolume([8.0, 8.0])   # explicit reference point
hv = front.hypervolume()             # reference auto-derived from the front's nadir
```

To **compare** two fronts, pass the *same* reference point to both. There is
also a standalone `pso.hypervolume(objectives, reference)` for fronts not
produced by `minimize_multi`. The metric uses the WFG algorithm, exact for any
number of objectives.

## From Rust

```rust
use turboswarm_core::prelude::*;

let space = ContinuousSpace::new(vec![(-5.0, 5.0); 2]);
let params = MopsoParams { seed: Some(42), ..Default::default() };
let res = Mopso::new(space, InertiaVelocity::new(0.729, 1.49445, 1.49445), params)
    .minimize(|x| vec![x.iter().map(|v| v * v).sum(),
                       x.iter().map(|v| (v - 2.0).powi(2)).sum()]);
// `res.front` is the non-dominated set: each solution has a decision vector
// with x ∈ [0, 2]² and its two objectives, tracing the trade-off front from
// (f1 ≈ 0, f2 ≈ 8) to (f1 ≈ 8, f2 ≈ 0).
for s in &res.front {
    println!("{:?} -> {:?}", s.position, s.objectives);
}
```
