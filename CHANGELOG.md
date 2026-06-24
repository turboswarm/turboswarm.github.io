# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.2.2] — 2026-06-24

### Added
- Reproducible benchmark suite (`benches/bench_suite.py`) comparing turboswarm
  against `pyswarms`, `pyswarm` and `pymoo` across functions and dimensions,
  with CSV results, machine provenance and a speedup figure.
- JOSS submission scaffolding: `paper.md`, `paper.bib`, `CITATION.cff` and
  `CONTRIBUTING.md`.

## [0.2.1] — 2026-06-24

### Fixed
- Packaging metadata: corrected the PyPI homepage link and added the
  Documentation and Enterprise URLs.
- Aligned the version across all sources (Rust workspace, `pyproject.toml` and
  `turboswarm.__version__`), which had drifted between 0.2.0 and 0.2.1.

## [0.2.0] — 2026-06-24

### Added
- **Grey numbers** — optimization over grey variables ⊗ = `[lower, upper]`
  (quantities known only to lie within an interval):
  - `GreySpace` (Rust): each grey variable is encoded internally as a
    center + spread pair, so the swarm coordinates stay decoupled. The whole
    decoded interval is kept within per-variable `(lower, upper)` limits by a
    coupled projection, with an optional extra `max_spread` cap on the
    half-width. `Grey` carries center/spread/lower/upper, whitenization
    (`whiten(λ)`) and interval arithmetic (`+`, `−`, `×`, scaling by a scalar).
  - `minimize_grey` (Python) returns a `GreyResult` (`best_position` as
    intervals, plus `best_centers`/`best_spreads`, `convergence`, …). The
    objective can be a callable or the name of a native grey benchmark.
    `representation="interval"` (default) or `"center_spread"` chooses how each
    grey number is passed to and read from a Python objective. `bounds` set the
    `(lower, upper)` limits each grey number must stay within.
  - `grey_sphere` benchmark (expected sphere + uncertainty penalty; optimum at
    the crisp origin) with metadata, dispatchable by name and exposed via
    `grey_benchmark_info`; mirrored in `turboswarm.benchmarks`.
  - Example notebook `notebooks/05_grey_numbers.ipynb`.
- **CI**: a `release-github` workflow creates the GitHub Release on every
  version tag, keeping the repo's "latest release" in sync with the version
  published to PyPI/crates.io.

## [0.1.3] — 2026-06-24

### Added
- **CEC-family benchmark functions**: `bent_cigar`, `discus`, `elliptic`
  (high-conditioned), `zakharov`, `levy` and `expanded_schaffer` (chained
  Schaffer F6) — the canonical base functions of the CEC suites (unshifted /
  unrotated; the official shift/rotation data is not bundled). Registered with
  metadata, exposed by name in `minimize`/`benchmark_info`, and mirrored in
  `turboswarm.benchmarks`.
- **Grid-based MOPSO archive** (Coello's adaptive hypercube grid): pass
  `grid_divisions=d` to `minimize_multi` (or set `MopsoParams.grid_divisions`) to
  keep the Pareto archive diverse with an adaptive grid of `d` cells per
  objective instead of the crowding-distance default — pruning drops members
  from the most crowded cell and leaders are drawn towards sparser cells.
- **Equality constraints and a repair operator** in `minimize`:
  `equality_constraints=` takes callables `h(x)` (feasible when `h(x) == 0`) and
  adds a quadratic penalty `penalty * sum(h(x)**2)`; `repair=` takes an operator
  `repair(x) -> x'` applied to each candidate before evaluation (and to the
  returned `best_position`, so the reported solution stays consistent). Both
  require a Python objective and are rejected on the native/vectorized paths.
- **Hyperparameter sensitivity analysis** (`turboswarm.sweep`): runs PSO over a
  Cartesian product of hyperparameter value lists (`grid={"w": [...], "c1":
  [...]}`), optionally repeated over several seeds, and returns a `SweepResult`
  (records with `mean`/`std`/`min`/`max` of the best value, `.best()`, and an
  optional `.to_dataframe()`). New `viz.plot_sensitivity` draws a line (1
  hyperparameter) or a heatmap (2).

## [0.1.2] — 2026-06-24

### Changed
- **Renamed the Rust crate `pso-core` → `turboswarm-core`** (module path
  `pso_core` → `turboswarm_core`) for brand consistency with the `turboswarm`
  Python package. The API is unchanged. The old `pso-core` 0.1.0/0.1.1 remain on
  crates.io; new Rust users should depend on `turboswarm-core`. The Python
  package `turboswarm` is unaffected (the rename is internal to the Rust side).

## [0.1.1] — 2026-06-24

### Added
- **Hypervolume** quality indicator for Pareto fronts
  (`pso_core::mopso::hypervolume`, the WFG algorithm — exact for any number of
  objectives). Exposed in Python as `ParetoFront.hypervolume(reference=None)`
  (auto reference from the front's nadir when omitted) and the standalone
  `turboswarm.hypervolume(front, reference)`.

## [0.1.0] — 2026-06-24

First public release: `turboswarm` on PyPI and `pso-core` on crates.io.

### Added
- Core foundations: the `SearchSpace`, `Velocity` and `Topology` traits and the
  generic PSO loop; `ContinuousSpace`/`IntegerSpace`; the inertia variant and
  global topology; history recording; sphere/rastrigin/rosenbrock benchmarks;
  Python bindings (`minimize`, `PsoResult`, `import turboswarm`) and the `viz`
  layer (convergence, comparison, 2D animation).
- **Constriction** velocity variant (Clerc-Kennedy): the χ factor is derived
  from `c1 + c2`.
- **FIPS** velocity variant (Fully Informed PSO): the particle is informed
  by its entire neighborhood, not just its best.
- **Ring** (`Ring`, lbest), **Von Neumann** (`VonNeumann`, toroidal 2D grid)
  and **Random** (`Random`, seeded random neighborhoods) topologies.
- **Early stopping** (`patience` + `tol`) and **velocity clamping** (`v_max`),
  exposed through both the Rust `PsoParams` and the Python `minimize`.
- **Inequality constraints** (penalty method) in the Python `minimize`
  (`constraints=`, `penalty=`); feasible when `g(x) <= 0`.
- **Parallel objective evaluation** from Rust: `Pso::minimize_parallel`
  (powered by `rayon`, synchronous updates) for expensive objectives.
- More **stop conditions**: target value (`target`), evaluation budget
  (`max_evals`) and wall-clock budget (`max_time`). `PsoResult` now reports
  `evaluations` and a `stop_reason`.
- **Per-iteration callback** (`callback=` in Python, `minimize_with_callback`
  in Rust): receives an `IterationInfo` snapshot and can request an early stop.
- **Boundary-handling strategies** (`bounds_handling=`): `clamp` (default),
  `reflect`, `wrap` and `reinit`, via `SearchSpace::enforce_bounds`.
- **Multi-objective optimization (MOPSO)**: `pso_core::mopso` (`Mopso`,
  `MopsoParams`, `MopsoResult`) and `minimize_multi` in Python return an
  approximated Pareto front (external archive + crowding distance, plus a
  turbulence/mutation operator via `mutation_rate`). Plus `viz.plot_pareto` to
  visualize a 2-objective front.
- **Mixed variable types**: `MixedSpace` (Rust) and `var_types=` (Python) allow
  a per-dimension choice of `real`, `integer` or `binary` in one problem.
- **Friendlier bounds**: `minimize`/`minimize_multi` accept either a list of
  per-dimension `(min, max)` pairs or a single `(min, max)` pair with `dim=N`.
  Rust adds `ContinuousSpace::uniform(dim, lo, hi)` and `IntegerSpace::uniform`.
- **Batched/vectorized objective**: `Pso::minimize_batch` in Rust and
  `vectorized=True` in Python evaluate the whole swarm per call. The Python path
  passes the swarm as a contiguous **NumPy array** (via the `numpy` crate) and
  reads results back as a slice — matching `pyswarms` on expensive vectorizable
  objectives.
- Documentation: a navigable MkDocs Material portal, including a `Comparison`
  page that benchmarks turboswarm against `pyswarms` and `pyswarm`.
- The package is named **turboswarm**.
- Example notebooks in `notebooks/` (quickstart; variants & topologies;
  integer/binary/mixed & constraints; multi-objective).

### Performance
- The main loop no longer clones the full neighborhood for variants that do not
  need it (only FIPS does), reuses scratch buffers across iterations, and
  updates best positions in place. This roughly halved the native run time,
  making `turboswarm` (native) ~3× faster than `pyswarms` on the benchmark
  functions. See the [Comparison](https://github.com/turboswarm/turboswarm.github.io) page.
- **ackley**, **griewank** and **schwefel** benchmarks (the latter with the optimum
  away from the origin), with metadata registration (`meta`/`ALL`).
- Python API: the new variants and topologies are selected by name
  (`velocity=`, `topology=`); new `benchmark_info(name)` function.
- Documentation: complete docstrings in Rust (`cargo doc`) and in the Python API
  (rendered with mkdocstrings); `#![warn(missing_docs)]` in `pso-core`; READMEs
  and publishing metadata (Cargo + PyPI).

### Changed
- The `Topology` trait is now defined by `neighbors(i)` (the full
  neighborhood); `best_neighbor` is derived by default. `UpdateContext` exposes
  `neighbor_bests` (the `pbest` of the entire neighborhood). The behavior of
  the classic variants does not change (verified by the determinism tests).
