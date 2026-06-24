# turboswarm-core

The Rust core of [**turboswarm**](https://github.com/turboswarm/turboswarm.github.io)
— a Particle Swarm Optimization library. Pure Rust, no FFI, zero-cost generics.

The PSO loop knows nothing about any concrete variant: everything that changes
lives behind three traits — `SearchSpace`, `Velocity` and `Topology`. Adding a
variant means implementing one trait, without touching the core.

## Features

- Velocity variants: inertia, constriction (Clerc-Kennedy) and FIPS.
- Topologies: global, ring, Von Neumann and random.
- Spaces: continuous, integer (with discretization) and mixed.
- Multi-objective optimization (MOPSO) in `turboswarm_core::mopso`.
- Parallel objective evaluation (`minimize_parallel`, via `rayon`).
- Run control: target / evaluation / time budgets, stagnation, callback.
- Benchmarks with metadata (sphere, rastrigin, rosenbrock, ackley, griewank,
  schwefel).

## Example

```rust
use turboswarm_core::prelude::*;
use turboswarm_core::benchmarks::rastrigin;

let space = ContinuousSpace::uniform(2, -5.12, 5.12);
let velocity = InertiaVelocity::new(0.729, 1.49445, 1.49445);
let params = PsoParams { seed: Some(42), ..Default::default() };

let result = Pso::new(space, velocity, GlobalBest::new(), params).minimize(rastrigin);
assert!(result.best_value < 1e-3);
```

The Python bindings live in the `turboswarm` package
(`pip install turboswarm`). Full docs: <https://turboswarm.github.io/>.

## License

MIT
