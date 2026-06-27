# Comparison with other libraries

How `turboswarm` compares to the most-used Python optimization libraries that
include PSO: [`pyswarms`](https://github.com/ljvmiranda921/pyswarms),
[`pyswarm`](https://pypi.org/project/pyswarm/),
[`pymoo`](https://pymoo.org) and [`DEAP`](https://github.com/DEAP/deap). The aim
is an honest picture — each leads somewhere.

## Feature comparison

| Feature | turboswarm | pyswarms | pyswarm | pymoo | DEAP |
|---|:--|:--|:--|:--|:--|
| Compute core | **Rust** (no GIL) | Python+NumPy | pure Python | Python+NumPy | pure Python |
| Velocity variants | inertia, constriction, **FIPS** | inertia, binary | inertia/constriction | inertia (adaptive) | recipe (you write it) |
| Topologies | global, ring, Von Neumann, random | global, ring, VN, pyramid, random | global | global | — |
| Real / integer / binary / mixed | ✅ all, one API | real, binary | real, integer | real, integer | recipe |
| Grey (interval) variables | ✅ | ❌ | ❌ | ❌ | ❌ |
| Constraints | ✅ penalty + repair | manual | ✅ inequality | ✅ rich | manual |
| Multi-objective | ✅ MOPSO | ❌ | ❌ | ✅ **many MOEAs** | ✅ NSGA-II etc. |
| Parallel evaluation | ✅ `rayon` (Rust) | ❌ | ❌ | ✅ | ✅ |
| Built-in visualization | convergence, compare, **animate 2D + 3D surface** | cost/contour/surface | ❌ | ✅ (scatter, PCP) | ❌ |
| Reproducible seed | ✅ deterministic | global NumPy seed | ✅ | ✅ | ✅ |
| Extensibility | ✅ one trait | backend (involved) | ❌ | ✅ operators | ✅ toolbox |
| Ecosystem integrations | SciPy, scikit-learn, Optuna, agents | — | — | — | — |
| API languages | **Rust + Python** | Python | Python | Python | Python |

**Where turboswarm leads:** a compiled Rust core (speed, below), FIPS +
constriction and a **grey/interval** search space out of the box, **2D and 3D
animated-swarm** visualization, deterministic seeding, a single API spanning real/integer/mixed
variables with a one-trait extensibility model, and first-class
[ecosystem integrations](guide/integrations.md) (a SciPy drop-in, `PSOSearchCV`,
an Optuna sampler, an agent tool).

**Where the others lead:** `pymoo` and `DEAP` are broad evolutionary-computation
frameworks — `pymoo` in particular has a large catalogue of multi-objective
algorithms and operators well beyond MOPSO; `DEAP` is a flexible general toolbox
(its PSO is an example recipe, not a packaged solver). `pyswarms` has one extra
topology, `pyramid` (Delaunay-based), which turboswarm omits on purpose (it needs
a computational-geometry dependency and rarely beats Von Neumann or ring).

## Performance

Wall-clock time to optimize standard functions, **identical configuration for
every library** (40 particles, 200 iterations, `w=0.729`, `c1=c2=1.49445`,
dim = 10), median of 5 seeds after a warm-up. `turboswarm` runs with
`record_history=False` so it does the same work as the others. Each library is
driven idiomatically: `pyswarms`/`pymoo` with a vectorized NumPy objective,
`pyswarm`/`DEAP`/`turboswarm (py)` with a scalar Python callable, and
`turboswarm (native)` with the benchmark computed in Rust.

| Function (dim 10) | turboswarm (native) | turboswarm (py) | pyswarms | pyswarm | pymoo | DEAP |
|---|--:|--:|--:|--:|--:|--:|
| sphere    | **4.6 ms** | 25.0 ms | 11.6 ms | 33.1 ms | 227.8 ms | 40.5 ms |
| rastrigin | **6.3 ms** | 52.0 ms | 14.6 ms | 57.6 ms | 231.9 ms | 57.3 ms |
| ackley    | **5.2 ms** | 50.4 ms | 14.1 ms | 66.2 ms | 226.8 ms | 66.1 ms |

!!! note "Read this honestly"
    - `turboswarm` (native) is the **fastest** across all functions and
      dimensions tested — roughly **2.5× faster than `pyswarms`** (vectorized
      NumPy), **~10× faster than `pyswarm`/`DEAP`** (pure-Python loops), and
      **~40× faster than `pymoo`** (whose per-generation framework overhead
      dominates at these sizes), at comparable solution quality.
    - The **`turboswarm` (py)** row — a plain *scalar* Python callable — is not
      faster than `pyswarms`: the per-evaluation GIL round-trip dominates and
      `pyswarms` vectorizes the objective in NumPy. For top speed use a native
      benchmark, call from **Rust**, or use the vectorized path.
    - With **`vectorized=True`** turboswarm **matches `pyswarms`** on expensive
      vectorizable objectives; for *cheap* objectives `pyswarms` is still
      ~1.5–2× faster, because it also vectorizes the swarm bookkeeping in NumPy.
    - `pymoo`/`DEAP` are not optimized for raw single-objective PSO speed — it is
      not their focus. Absolute numbers are **machine-dependent**; the
      within-run ratios are the meaningful part. Measured on an Apple-silicon
      laptop; reproduce on your hardware.

## Statistical comparison

Treating each (function, dimension) pair as a problem instance (15 in total) and
the libraries as the methods (the standard Demšar setup), `benches/stats.py`
reports per-library **mean ranks**, the **Friedman** test and pairwise
**Wilcoxon** tests — for both solution quality and time.

- **Solution quality** (best value): the Friedman test finds **no significant
  difference** (p ≈ 0.13). All libraries reach comparable quality; mean ranks
  are close (DEAP and `turboswarm` lead slightly). This is the honest result —
  on these standard functions nobody clearly wins on quality.
- **Wall-clock time:** the difference is **highly significant** (Friedman
  p ≈ 6e-14). `turboswarm (native)` has mean rank **1.00** — fastest on *every*
  one of the 15 instances — and a pairwise Wilcoxon vs each other library gives
  p ≈ 6e-5 (the smallest attainable with 15 instances).

In short: **comparable quality, decisively faster.** Reproduce with
`python benches/stats.py` (after `bench_suite.py`).

## Hyperparameter search

For hyperparameter tuning specifically, turboswarm integrates where you already
work instead of competing head-on:

- [`PSOSearchCV`](guide/integrations.md#scikit-learn) is a drop-in alternative to
  scikit-learn's `GridSearchCV` / `RandomizedSearchCV` that searches the
  **continuous** space the grid can only sample.
- [`TurboswarmSampler`](guide/integrations.md#optuna) plugs PSO into an **Optuna**
  study as a sampler, keeping Optuna's storage, pruning and dashboards.

See the [hyperparameter-tuning tutorial](tutorials/hyperparameter-tuning.md) for
a worked `PSOSearchCV`-vs-`GridSearchCV` comparison.

## Reproduce it

The benchmark suite covers five functions × three dimensions × all libraries,
with machine provenance:

```bash
pip install -e . pyswarms pyswarm pymoo deap
python benches/bench_suite.py              # plain text
python benches/bench_suite.py --markdown   # Markdown table
python benches/stats.py                     # Friedman / Wilcoxon / ranking
```

It writes `benches/results/results.csv`, `meta.json` (machine + versions) and a
speedup figure. Source:
[`benches/bench_suite.py`](https://github.com/turboswarm/turboswarm.github.io/blob/main/benches/bench_suite.py).
