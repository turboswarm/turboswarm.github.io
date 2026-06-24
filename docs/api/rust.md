# Rust API

The Rust core (`turboswarm-core`) is documented with rustdoc. It is not embedded in
this portal; build it locally or browse it on docs.rs once published.

## Build locally

```bash
cargo doc -p turboswarm-core --no-deps --open
```

This opens `target/doc/turboswarm_core/index.html`.

## Published

Once `turboswarm-core` is published to crates.io, the API will be available at
[docs.rs/turboswarm-core](https://docs.rs/turboswarm-core).

## Map of the core

| Item | Path | Role |
|------|------|------|
| `Pso` | `turboswarm_core::pso::Pso` | The optimizer / main loop |
| `PsoParams` | `turboswarm_core::params::PsoParams` | Run parameters |
| `PsoResult` | `turboswarm_core::pso::PsoResult` | Result with history |
| `SearchSpace` | `turboswarm_core::traits::SearchSpace` | Domain trait |
| `Velocity` | `turboswarm_core::traits::Velocity` | Velocity-rule trait |
| `Topology` | `turboswarm_core::traits::Topology` | Social-structure trait |
| `ContinuousSpace`, `IntegerSpace` | `turboswarm_core::spaces` | Spaces |
| `InertiaVelocity`, `ConstrictionVelocity`, `FipsVelocity` | `turboswarm_core::velocity` | Variants |
| `GlobalBest`, `Ring`, `VonNeumann` | `turboswarm_core::topology` | Topologies |
| `benchmarks` | `turboswarm_core::benchmarks` | Test functions + metadata |

Everything is re-exported through `turboswarm_core::prelude` for convenient use.
