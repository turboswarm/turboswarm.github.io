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
| `integer`, `binary`, `var_types` | same as in `minimize` (mixed problems supported) |

## Visualizing the front

For two objectives:

```python
import matplotlib.pyplot as plt
pso.viz.plot_pareto(front)
plt.show()
```

## From Rust

```rust
use pso_core::prelude::*;

let space = ContinuousSpace::new(vec![(-5.0, 5.0); 2]);
let params = MopsoParams { seed: Some(42), ..Default::default() };
let res = Mopso::new(space, InertiaVelocity::new(0.729, 1.49445, 1.49445), params)
    .minimize(|x| vec![x.iter().map(|v| v * v).sum(),
                       x.iter().map(|v| (v - 2.0).powi(2)).sum()]);
for s in &res.front {
    println!("{:?} -> {:?}", s.position, s.objectives);
}
```
