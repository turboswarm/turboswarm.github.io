# Contributing to turboswarm

Thanks for your interest in contributing! This project is a general-purpose
Particle Swarm Optimization library with a Rust core and a Python API.
Contributions of all kinds are welcome: bug reports, documentation, new
benchmarks, new PSO variants and topologies, and performance work.

## Reporting bugs and requesting features

Please open an issue at
<https://github.com/turboswarm/turboswarm.github.io/issues> and include, where
relevant:

- what you expected to happen and what happened instead;
- a minimal reproducible example (Python or Rust);
- your OS, Rust version (`rustc --version`) and Python version;
- the `turboswarm` / `turboswarm-core` version.

## Development setup

You need [Rust](https://rustup.rs) (tested with 1.75+) and Python ≥ 3.9.

```bash
# Rust core
cargo build
cargo test -p turboswarm-core      # convergence tests + doctest
cargo clippy
cargo fmt

# Python (Rust core via maturin)
python -m venv .venv && source .venv/bin/activate
pip install maturin matplotlib numpy
maturin develop --release    # compiles the Rust core and installs it
python examples/quickstart.py
```

After **any** change in `crates/turboswarm-core` or `crates/pso-py` you must
re-run `maturin develop` so the Python layer sees the updated core.

## Project layout

- `crates/turboswarm-core/` — Rust core (zero-cost generics, no FFI).
- `crates/pso-py/` — PyO3 bindings (native module `turboswarm_native`).
- `python/turboswarm/` — Python API, pure benchmarks and visualization.
- `examples/`, `notebooks/`, `benches/` — examples, notebooks and benchmarks.

The optimization loop knows nothing about any concrete variant. Everything that
changes between variants lives behind three traits in
`crates/turboswarm-core/src/traits.rs`: `SearchSpace`, `Velocity` and
`Topology`. See the "How to extend" section of
[`CLAUDE.md`](CLAUDE.md) for step-by-step recipes.

## Adding a PSO variant or topology

1. Implement the relevant trait in `velocity/` or `topology/` (use
   `velocity/inertia.rs` / `topology/global.rs` as templates).
2. Export it in the corresponding `mod.rs`.
3. Add a **convergence test** against a known optimum
   (`crates/turboswarm-core/tests/convergence.rs`).
4. Expose it by name in the binding (`build_velocity` / `build_topology` in
   `crates/pso-py/src/lib.rs`).
5. Run `maturin develop` and test from Python.

## Adding a benchmark

1. Add the function and its `Benchmark` metadata in
   `crates/turboswarm-core/src/benchmarks/functions.rs`.
2. Export it in `benchmarks/mod.rs`.
3. Add it to the `match` in `native_benchmark` in the binding.
4. (Optional) mirror it in `python/turboswarm/benchmarks.py`.

## Coding conventions

- **English everywhere** — comments, documentation, identifiers and prose. The
  project is published internationally on GitHub, PyPI and crates.io.
- **Reproducibility** — every experiment accepts a `seed`; keep determinism with
  a fixed seed, as the tests depend on it.
- Run `cargo fmt` and `cargo clippy` before submitting; CI enforces both.
- Every new variant or function needs a convergence test against its known
  optimum.

## Pull requests

1. Fork the repository and create a topic branch.
2. Make your change with accompanying tests and documentation.
3. Ensure `cargo fmt`, `cargo clippy`, `cargo test -p turboswarm-core` and the
   Python tests pass locally.
4. Open a pull request describing the motivation and the change. CI
   ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs formatting,
   linting and tests on every push and PR.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) of the project.
