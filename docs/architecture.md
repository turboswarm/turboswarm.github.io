# Architecture

The design goal is extensibility: you can add a PSO variant by implementing one
trait, without reading or touching the core loop.

## The crates

```
crates/turboswarm-core/   Rust core. No FFI dependencies. Zero-cost generics.
crates/pso-py/     PyO3 bindings (native module `turboswarm_native`).
python/turboswarm/     Python package: API, pure benchmarks, viz (matplotlib).
```

## Three traits, one loop

The PSO loop in `crates/turboswarm-core/src/pso.rs` knows nothing about any concrete
variant. Everything that changes lives behind three traits in `traits.rs`:

- **`SearchSpace`** — the problem domain: dimension, bounds, sampling, clamping
  and `decode`. **The integer/real difference lives only in `decode`**, which
  translates the internal continuous representation (`Vec<f64>`, always) into
  the evaluable type (`f64` or `i64`).
- **`Velocity`** — the velocity update rule. **One variant = one impl.** It
  receives an `UpdateContext` with the current state, the neighborhood best,
  and `neighbor_bests` (the personal bests of the whole neighborhood, used by
  fully informed variants like FIPS).
- **`Topology`** — the social structure. Defined by `neighbors(i)` (the
  neighborhood, including the particle itself); `best_neighbor` is derived from
  it by default.

## Key invariant

All positions and velocities are `Vec<f64>` inside the optimizer. The loop
never introduces generics over the position type — that is what keeps the FFI
boundary simple. The continuous-to-integer step is confined to
`SearchSpace::decode`.

## The Rust ↔ Python boundary

The core uses generics (zero-cost) for Rust use. For Python the variant and
topology are chosen at runtime by string, so `traits.rs` implements `Velocity`
and `Topology` for `Box<dyn ...>`. That lets
`Pso<S, Box<dyn Velocity>, Box<dyn Topology>>` reuse the exact same loop without
duplication.

The binding (`crates/pso-py/src/lib.rs`) exposes `minimize(...)`, which accepts
either a Python callable `f(list) -> float` (re-acquires the GIL per evaluation)
or the name of a native Rust benchmark (runs without the GIL).
