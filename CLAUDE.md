# CLAUDE.md

Guide for working in this repository with Claude Code.

## What this project is

A general-purpose **Particle Swarm Optimization (PSO)** library with a **compute
core in Rust** and a **Python API**, published on PyPI and crates.io. Priorities,
in order: 1) visualization/animation, 2) algorithm comparison, 3) code clarity,
4) performance.

It supports **real, integer and mixed** variables, and is designed as an
**extensible framework**: adding a PSO variant = implementing a trait, without
touching the core.

## Architecture

```
crates/turboswarm-core/   Rust core. No FFI dependencies. Zero-cost generics.
crates/pso-py/     PyO3 bindings (native module `turboswarm_native`).
python/turboswarm/     Python package: API (__init__), pure benchmarks, viz (matplotlib).
examples/          Rust (in turboswarm-core) and Python (quickstart.py) examples.
notebooks/         Example notebooks (quickstart, variants, integer, MOPSO).
```

### The central design (read before touching the core)

The PSO loop (`crates/turboswarm-core/src/pso.rs`) does NOT know any concrete variant.
Everything that changes between variants lives behind three traits in
`crates/turboswarm-core/src/traits.rs`:

- `SearchSpace` — the domain. **The integer/real difference lives only in `decode`**,
  which translates the internal continuous representation (`Vec<f64>`, always) to the
  evaluable type (`f64` or `i64`). Discretization happens only at evaluation time.
- `Velocity` — the velocity update rule. **One variant = one impl
  of this trait.** Template: `velocity/inertia.rs`. It receives an `UpdateContext`
  with `neighbor_best` (the neighborhood best) and `neighbor_bests` (pbest of
  the ENTIRE neighborhood; used by fully informed variants such as FIPS).
- `Topology` — the social structure of the swarm. **Fundamental method:
  `neighbors(i) -> Vec<usize>`** (neighborhood, includes the particle itself);
  `best_neighbor` is derived by default. Template: `topology/global.rs`.

**Important invariant:** all positions/velocities are `Vec<f64>` inside
the optimizer. Do not introduce generics over the position type in the loop.

### Rust↔Python boundary (important)

The core uses generics (zero-cost) for use from Rust. For Python, the
variants are selected at runtime by string, so `traits.rs` implements
`Velocity` and `Topology` for `Box<dyn ...>`. This enables
`Pso<S, Box<dyn Velocity>, Box<dyn Topology>>` without duplicating the loop.
**Do not break those `Box<dyn ...>` impls**: the binding depends on them.

The binding (`crates/pso-py/src/lib.rs`) exposes `minimize(...)`, which accepts:
- a Python callable `f(list) -> float` (reacquires the GIL per evaluation), or
- the name of a native Rust benchmark (runs without the GIL).

`i64 -> f64` is done with the local `ToF64` trait (do not use `Into<f64>`, it does
not exist for `i64`).

## Commands (verified)

### Rust
```bash
cargo build                          # compiles the whole workspace
cargo test -p turboswarm-core               # convergence tests + doctest
cargo run --example basic -p turboswarm-core
cargo clippy                         # linting (configured in .vscode)
```

### Python (Rust core via maturin)
```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin matplotlib numpy
maturin develop --release            # compiles the Rust core and installs it
python examples/quickstart.py
```

After ANY change in `crates/turboswarm-core` or `crates/pso-py`, you must re-run
`maturin develop` so that Python sees the changes.

### Toolchain
Requires Rust (rustup.rs) and Python ≥3.9. Tested with Rust 1.75 + Python 3.12 +
PyO3 0.22.

## Conventions

- **Language:** all comments, documentation, identifiers, and prose are in
  English. This project is published internationally (GitHub and PyPI), so keep
  everything in English.
- **Reproducibility:** every experiment accepts a `seed`. Maintain determinism
  with a fixed seed; the tests depend on it.
- **History:** `record_history` defaults to `true` (visualization is the
  top priority). In production it would be disabled, but not here.
- **Tests:** every new variant or function needs a convergence test
  against its known optimum (see `crates/turboswarm-core/tests/convergence.rs`).
- **Benchmarks:** when adding a test function, also register its
  metadata (`Benchmark`) with its optimum, and expose it by name in the binding
  (`native_benchmark` in `crates/pso-py/src/lib.rs`).

## How to extend (typical tasks)

### Add a PSO variant
1. Create `crates/turboswarm-core/src/velocity/<name>.rs` implementing `Velocity`.
2. Export it in `velocity/mod.rs`.
3. Add a convergence test.
4. Expose it by name in `build_velocity` (`crates/pso-py/src/lib.rs`).
5. `maturin develop` and test from Python.

### Add a topology
Same as above, in `topology/`, implementing `Topology`, and exposing it in
`build_topology`.

### Add a benchmark
1. Function + `Benchmark` metadata in `crates/turboswarm-core/src/benchmarks/functions.rs`.
2. Export in `benchmarks/mod.rs`.
3. Add to the `match` of `native_benchmark` in the binding.
4. (Optional) mirror in `python/turboswarm/benchmarks.py`.

## Status by phase

- ✅ **Phase 1** — Rust core: continuous + integer space, inertia variant,
  global topology, history, benchmarks (sphere/rastrigin/rosenbrock), tests.
- ✅ **Phase 2** — Variants: inertia + **constriction** (Clerc-Kennedy) +
  **FIPS** (fully informed). Topologies: global + **ring** (`Ring`) +
  **Von Neumann** (`VonNeumann`). Benchmarks: + **ackley, griewank, schwefel**
  (with `meta`/`ALL` registration and `benchmark_info` in Python). All exposed by
  name in the binding and tested from Python.
- ✅ **Phase 3** — Python API (`import turboswarm`), real/integer space, Python
  function or native benchmark, `PsoResult` with convergence and history.
- ✅ **Phase 4** — `turboswarm.viz` (convergence, comparison, 2D animation,
  Pareto plot) and example notebooks in `notebooks/`.
- ✅ **Phase 5** — `Random` topology; `MixedSpace` (per-dimension real/integer/
  binary) + `binary`/`var_types`; inequality **constraints** (penalty); run
  control (`target`/`max_evals`/`max_time`/`patience`, `stop_reason`,
  `evaluations`, per-iteration callback); **boundary handling**
  (clamp/reflect/wrap/reinit); `v_max`; **parallel** (`minimize_parallel`,
  rayon) and **vectorized** (`minimize_batch` / `vectorized=True`, NumPy).
- ✅ **Phase 6** — Multi-objective **MOPSO** in `turboswarm_core::mopso` (`Mopso`,
  Pareto archive + crowding + turbulence); `minimize_multi` → `ParetoFront`.

See `ROADMAP.md` for the breakdown of checkable tasks.

## Known pitfalls

- After editing Rust, remember `maturin develop` before testing in Python.
- `animate_swarm` only supports 2D problems and requires `record_history=True`.
- Do not add generics over the position type in the PSO loop: it would break the
  FFI boundary. The integer/real difference always goes in `SearchSpace::decode`.
- The doctest of `lib.rs` runs in `cargo test`: keep it valid.
