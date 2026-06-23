# Installation

The library has two faces: a pure-Rust core (`pso-core`) usable from Rust, and
a Python package (`turboswarm`) whose native module is built from the same core.

## Requirements

- **Rust** (via [rustup](https://rustup.rs)) — for the core and the build.
- **Python ≥ 3.9** — for the Python API.

!!! warning "Python version and PyO3"
    PyO3 0.22 supports up to Python 3.13. If your system Python is newer
    (e.g. 3.14), build the wheel against a supported interpreter such as
    Python 3.12.

## Python (development build with maturin)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install maturin matplotlib numpy
maturin develop --release      # compiles the Rust core and installs turboswarm
```

After **any** change in `crates/pso-core` or `crates/pso-py`, re-run
`maturin develop` so Python picks up the changes.

Verify it:

```bash
python -c "import turboswarm; print('ok')"
python examples/quickstart.py
```

## Rust only

```bash
cargo build                          # whole workspace
cargo test -p pso-core               # convergence tests + doctest
cargo run --example basic -p pso-core
```

## Optional extras

```bash
pip install -e ".[docs]"        # build this documentation portal (mkdocs)
pip install -e ".[notebooks]"   # jupyter + plotly
```
