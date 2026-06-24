# Extending

Adding to `pso-rs` means implementing a trait in the core and (optionally)
exposing it by name in the Python binding. No changes to the PSO loop.

## Add a velocity variant

1. Create `crates/turboswarm-core/src/velocity/<name>.rs` implementing `Velocity`.
2. Export it in `velocity/mod.rs`.
3. Add a convergence test (see `crates/turboswarm-core/tests/convergence.rs`).
4. Expose it by name in `build_velocity` (`crates/pso-py/src/lib.rs`).
5. Run `maturin develop` and try it from Python.

A minimal variant only needs the `update` method:

```rust
use rand::{Rng, RngCore};
use turboswarm_core::traits::{UpdateContext, Velocity};

pub struct MyVelocity { pub w: f64 }

impl Velocity for MyVelocity {
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64> {
        let mut v = Vec::with_capacity(ctx.position.len());
        for d in 0..ctx.position.len() {
            let r: f64 = rng.gen();
            v.push(self.w * ctx.velocity[d]
                   + r * (ctx.neighbor_best[d] - ctx.position[d]));
        }
        v
    }
}
```

`UpdateContext` also exposes `neighbor_bests` (every neighbor's personal best),
`iteration` and `max_iterations` for fully informed or scheduled variants.

## Add a topology

Same shape, under `topology/`, implementing `Topology`. You only need
`neighbors`; `best_neighbor` is derived for you:

```rust
use turboswarm_core::swarm::Swarm;
use turboswarm_core::traits::Topology;

pub struct MyTopology;

impl Topology for MyTopology {
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize> {
        // Return the indices that inform particle `i` (include `i` itself).
        (0..swarm.len()).collect()
    }
}
```

Expose it in `build_topology` in the binding.

## Add a benchmark

1. Function + `Benchmark` metadata in `crates/turboswarm-core/src/benchmarks/functions.rs`.
2. Export it in `benchmarks/mod.rs` (and add it to `ALL`).
3. Add it to the `match` in `native_benchmark` in the binding.
4. (Optional) mirror it in `python/turboswarm/benchmarks.py`.

## Conventions

- All comments, documentation and identifiers are in English.
- Every new variant or function needs a convergence test against its known
  optimum.
- Keep runs reproducible with a fixed `seed`; the tests rely on it.
