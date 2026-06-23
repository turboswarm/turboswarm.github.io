# Comparison with other PSO libraries

How `turboswarm` compares to the two most-used Python PSO libraries,
[`pyswarms`](https://github.com/ljvmiranda921/pyswarms) and
[`pyswarm`](https://pypi.org/project/pyswarm/). The aim is an honest picture:
each library has areas where it leads.

## Feature comparison

| Feature | turboswarm | pyswarms | pyswarm |
|---------|:----------:|:--------:|:-------:|
| Compute core | **Rust** (native, no GIL) | Python + NumPy | Pure Python |
| Velocity variants | inertia, constriction, **FIPS** | inertia (global/local/general), binary | inertia/constriction |
| Topologies | global, ring, Von Neumann, random | global, ring, Von Neumann, pyramid, random | global |
| Real variables | ✅ | ✅ | ✅ |
| Integer variables | ✅ (decode + discretization) | binary only | ✅ (`intvar`) |
| Binary variables | ✅ (`binary=True`) | ✅ (BinaryPSO) | ✅ (`intvar`) |
| Constraints | ✅ (inequality, penalty) | manual penalty | ✅ (inequality) |
| Parallel evaluation | ✅ (`rayon`, from Rust) | ❌ | ❌ |
| Multi-objective (MOPSO) | ✅ (`minimize_multi`) | ❌ | ❌ |
| Early stopping | ✅ (`tol` + `patience`) | ✅ (`ftol`) | ✅ (`minfunc`/`patience`) |
| Velocity clamp | ✅ (`v_max`) | ✅ | ❌ |
| Built-in visualization | convergence, compare, **animate 2D** | cost history, contour, surface | ❌ |
| Reproducible seed | ✅ (built-in, deterministic) | via global NumPy seed | ✅ |
| Custom variant (extensibility) | ✅ (implement one trait) | ✅ (backend, more involved) | ❌ |
| API languages | **Rust + Python** | Python | Python |
| License | MIT | MIT | BSD-3 |

**Where turboswarm leads:** a Rust core, the only one with FIPS + constriction
out of the box, **multi-objective optimization (MOPSO)**, parallel evaluation,
animated-swarm visualization, deterministic seeding, and a single API for real,
integer *and* mixed variables with a one-trait extensibility model.

**Where the others lead:** `pyswarms` has one extra topology — `pyramid`
(Delaunay-based). turboswarm omits it on purpose: it needs a computational-
geometry dependency and rarely beats Von Neumann or ring in practice. That is
the only remaining gap.

## Performance

Wall-clock time to optimize standard functions, **identical configuration for
every library** (40 particles, 200 iterations, `w=0.729`, `c1=c2=1.49445`,
dim = 10), median of 5 runs. `turboswarm` runs with `record_history=False` so
it does the same work as the others.

Each library is used **idiomatically**: `pyswarms` with a vectorized NumPy
objective (its intended usage), `pyswarm` and the `turboswarm (py)` row with a
scalar Python callable, and `turboswarm (native)` with the benchmark computed
in Rust.

| Function | turboswarm (native) | turboswarm (py) | pyswarms (vectorized) | pyswarm |
|----------|--------------------:|----------------:|----------------------:|--------:|
| sphere    | **~6.5 ms** | ~50 ms | ~21 ms | ~60 ms |
| rastrigin | **~7 ms**   | ~80 ms | ~24 ms | ~105 ms |
| ackley    | **~8 ms**   | ~88 ms | ~23 ms | ~120 ms |

!!! note "Read this honestly"
    - `turboswarm` (native) is the **fastest** here — roughly **3× faster than
      `pyswarms`** (vectorized NumPy) and **~10× faster than `pyswarm`**, at
      equal or better solution quality.
    - This relies on the Rust core *and* a lean loop: classic variants no longer
      clone the whole neighborhood (only FIPS needs it) and scratch buffers are
      reused. That roughly **halved** the native time.
    - The **`turboswarm` (py)** row — a plain *scalar* Python callable — is not
      faster than `pyswarms`: there, the per-evaluation GIL round-trip dominates
      and `pyswarms` vectorizes the objective in NumPy. It lands in `pyswarm`'s
      ballpark. For top speed use a native benchmark, call from **Rust**, or use
      the vectorized path below.
    - With **`vectorized=True`** (the swarm passed as one NumPy array per
      iteration), turboswarm **matches `pyswarms` for expensive vectorizable
      objectives** (measured ratio ≈ 1.0). For *cheap* objectives `pyswarms` is
      still ~1.5–2× faster, because it also vectorizes the swarm bookkeeping in
      NumPy while turboswarm updates the swarm in Rust per particle.
    - Absolute numbers and ratios are **machine- and load-dependent**; the
      within-run ratios are the meaningful part. Reproduce on your hardware.

### Parallel evaluation (expensive objectives)

For costly objectives, the Rust API offers `Pso::minimize_parallel`, which
evaluates the swarm in parallel with `rayon`. On a 10-core machine, an
artificially expensive objective runs **~6.8× faster** than the sequential path
(`cargo run --release --example parallel -p pso-core`). It uses synchronous
(Jacobi) updates and needs an `Fn + Sync` objective, so it is a Rust-side
feature — the Python-callable path is serialized by the GIL and cannot benefit.


Numbers are machine-dependent — always reproduce on your own hardware.

## Reproduce it

```bash
pip install -e ".[docs]" pyswarms pyswarm
python scripts/bench_vs_libs.py            # plain text
python scripts/bench_vs_libs.py --markdown # Markdown table
```

The benchmark script is [`scripts/bench_vs_libs.py`](https://github.com/joselsalmeron/turboswarm/blob/main/scripts/bench_vs_libs.py).
