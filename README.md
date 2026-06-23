# turboswarm

A general-purpose, extensible **Particle Swarm Optimization (PSO)** library with
a **Rust** core, valid for **real, integer and mixed** variables. Its design
priorities are visualization, algorithm comparison and code clarity.

Usable directly from **Rust** and from **Python** (via PyO3 + maturin).

## What it includes

- **Velocity variants:** inertia (Shi-Eberhart), constriction (Clerc-Kennedy)
  and FIPS (Fully Informed PSO).
- **Topologies:** global (gbest), ring (lbest), Von Neumann (2D grid) and random.
- **Spaces:** continuous (real), integer (with a discretization strategy),
  binary and **mixed** (per-dimension type).
- **Multi-objective (MOPSO):** Pareto front via an external archive + crowding
  distance, with a turbulence operator.
- **Constraints:** inequality constraints via a penalty.
- **Run control:** stop on target value, evaluation budget, wall-clock budget
  or stagnation; per-iteration callback; the result reports `stop_reason` and
  `evaluations`.
- **Boundary handling:** clamp, reflect, wrap or reinit.
- **Performance:** velocity clamp (`v_max`), parallel evaluation (`rayon`) and a
  vectorized/batched objective path.
- **Benchmarks:** sphere, rastrigin, rosenbrock, ackley, griewank and schwefel
  (with metadata: recommended bound and known optimum).
- **Visualization** (Python): convergence curves, variant comparison, 2D swarm
  animation and Pareto-front plots.
- **Reproducibility:** every experiment accepts a `seed`.

## Status by phase

| Phase | Content | Status |
|------|-----------|--------|
| 1 | Rust core: continuous+integer spaces, inertia, gbest, history, benchmarks, tests | ✅ |
| 2 | More variants (constriction, FIPS), topologies (ring, Von Neumann, random), benchmarks | ✅ |
| 3 | Python bindings (PyO3 + maturin) — `import turboswarm` API | ✅ |
| 4 | Visualization (`viz`) + example notebooks | ✅ |
| 5 | Constraints, mixed variables, run control, boundary handling, parallel/vectorized | ✅ |
| 6 | Multi-objective optimization (MOPSO) | ✅ |

See [`ROADMAP.md`](ROADMAP.md) for the task breakdown.

## Quick start (Rust)

```bash
cargo test -p pso-core           # runs the suite (convergence tests + doctest)
cargo run --example basic -p pso-core
```

```rust
use pso_core::prelude::*;
use pso_core::benchmarks::rastrigin;

let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
let velocity = InertiaVelocity::new(0.9, 1.49445, 1.49445).with_decay(0.4);
let params = PsoParams { seed: Some(42), ..Default::default() };

let result = Pso::new(space, velocity, GlobalBest::new(), params)
    .minimize(rastrigin);

println!("{:?} -> {}", result.best_position, result.best_value);
```

## Quick start (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin matplotlib numpy
maturin develop --release        # compiles the Rust core and installs it
python examples/quickstart.py
```

```python
import turboswarm as pso

# Native benchmark (runs in Rust, without the GIL)
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=42)

# Variant and topology by name
r = pso.minimize("ackley", bounds=[(-32.768, 32.768)] * 2,
                 velocity="fips", topology="ring", seed=1)

# Your own function, or integer variables
r = pso.minimize(lambda x: sum(xi**2 for xi in x), bounds=[(-5, 5)] * 3)
r = pso.minimize(f, bounds=[(-10, 10)] * 2, integer=True)

print(r.best_position, r.best_value)
```

## The design idea (extensibility)

The PSO loop knows nothing about any variant. Everything that changes lives behind
three traits in [`crates/pso-core/src/traits.rs`](crates/pso-core/src/traits.rs):

- `SearchSpace` — the domain; the integer/real difference lives here (`decode`).
- `Velocity` — the update rule; **one variant = one impl** of this
  trait. It receives an `UpdateContext` with the neighborhood best and, for fully
  informed variants (FIPS), the `pbest` of the entire neighborhood.
- `Topology` — the social structure of the swarm. It is defined by its
  `neighbors(i)` method; `best_neighbor` is derived by default.

To create a new variant, implement `Velocity` and expose it by name in
the binding. See `velocity/inertia.rs` as a template and the "How to
extend" section of [`CLAUDE.md`](CLAUDE.md).

## Documentation

A navigable documentation portal (narrative guide + Python API reference) is
built with MkDocs Material and published to **GitHub Pages**:
<https://turboswarm.github.io/> *(live once Pages is enabled)*.

Build it locally:

```bash
pip install -e ".[docs]"          # mkdocs-material + mkdocstrings
./scripts/build-docs.sh --serve   # live portal at http://127.0.0.1:8000
./scripts/build-docs.sh           # build to site/ and the Rust API to target/doc/
```

The narrative sources live in [`docs/`](docs/); the Python API is generated
from docstrings via `mkdocstrings`. The Rust API is generated separately with
rustdoc (`cargo doc -p pso-core --no-deps --open`) and, once published, will be
available on docs.rs.

**Deployment:** [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
builds the package (so `mkdocstrings` can import it) and deploys on every push
to `main`. Enable it once in **Settings → Pages → Source: GitHub Actions**.

## Structure

```
crates/pso-core/   Rust core (zero-cost generics, no FFI)
crates/pso-py/     PyO3 bindings (native module turboswarm_native)
python/turboswarm/     Python API: __init__, pure benchmarks, viz (matplotlib)
notebooks/         example notebooks
examples/          Rust (basic) and Python (quickstart.py) examples
```

## License

MIT. See [`LICENSE`](LICENSE).
