# Rust API

The Rust core (`pso-core`) is documented with rustdoc. It is not embedded in
this portal; build it locally or browse it on docs.rs once published.

## Build locally

```bash
cargo doc -p pso-core --no-deps --open
```

This opens `target/doc/pso_core/index.html`.

## Published

Once `pso-core` is published to crates.io, the API will be available at
[docs.rs/pso-core](https://docs.rs/pso-core).

## Map of the core

| Item | Path | Role |
|------|------|------|
| `Pso` | `pso_core::pso::Pso` | The optimizer / main loop |
| `PsoParams` | `pso_core::params::PsoParams` | Run parameters |
| `PsoResult` | `pso_core::pso::PsoResult` | Result with history |
| `SearchSpace` | `pso_core::traits::SearchSpace` | Domain trait |
| `Velocity` | `pso_core::traits::Velocity` | Velocity-rule trait |
| `Topology` | `pso_core::traits::Topology` | Social-structure trait |
| `ContinuousSpace`, `IntegerSpace` | `pso_core::spaces` | Spaces |
| `InertiaVelocity`, `ConstrictionVelocity`, `FipsVelocity` | `pso_core::velocity` | Variants |
| `GlobalBest`, `Ring`, `VonNeumann` | `pso_core::topology` | Topologies |
| `benchmarks` | `pso_core::benchmarks` | Test functions + metadata |

Everything is re-exported through `pso_core::prelude` for convenient use.
