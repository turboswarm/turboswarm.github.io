# Roadmap

Check off each task as you progress. Phase 1 is already done and verified.

## ✅ Phase 1 — Minimal functional core (DONE)
- [x] Cargo workspace + `turboswarm-core` crate
- [x] Central traits: `SearchSpace`, `Velocity`, `Topology`
- [x] `ContinuousSpace` and `IntegerSpace` (with discretization strategies)
- [x] `InertiaVelocity` variant (with optional decay)
- [x] `GlobalBest` topology
- [x] History recording (`History`)
- [x] Benchmarks: sphere, rastrigin, rosenbrock
- [x] Convergence + reproducibility tests
- [x] Runnable `basic` example

## ✅ Phase 2 — Extensibility (DONE)
- [x] `ConstrictionVelocity` (Clerc-Kennedy). `velocity/constriction.rs`,
      χ derived from c1+c2. Exposed as `"constriction"` in the binding.
- [x] `FipsVelocity` (Fully Informed PSO). `velocity/fips.rs`, exposed as
      `"fips"`. It required extending the core cleanly: `Topology` is now
      defined by `neighbors()` (the full neighborhood) and `best_neighbor` is
      derived by default; `UpdateContext` carries `neighbor_bests` (pbest of the
      entire neighborhood) and the loop collects it. Behavior of the classic
      variants unchanged (verified by the determinism tests).
- [x] `Ring` topology (ring, `2·each_side+1` neighborhood). `topology/ring.rs`.
      Exposed as `"ring"`.
- [x] `VonNeumann` topology (toroidal 2D grid). `topology/von_neumann.rs`.
      Exposed as `"vonneumann"`; `square_for(n)` sizes the grid.
- [x] `Random` topology (seeded random neighborhoods). `topology/random.rs`.
      Exposed as `"random"`.
- [x] More benchmarks: ackley, griewank, schwefel (+ `meta`/`ALL` registration and
      `benchmark_info(name)` exposed in Python).
- [x] Tests comparing variants/topologies on the same functions and seed.
- [x] Configurable velocity limit (`v_max`).
- [x] Early stopping criteria (`patience` + `tol`).

## ✅ Phase 5 — Reaching (and passing) feature parity with other libraries
- [x] Inequality constraints (penalty method) in the Python `minimize`
      (`constraints=`, `penalty=`).
- [x] Parallel objective evaluation from Rust (`minimize_parallel`, `rayon`).
- [x] Binary optimization helper (`binary=True` in the Python `minimize`).
- [~] `Pyramid` topology (Delaunay-based) — **WON'T DO**. Architecturally
      possible (the `Topology` trait already receives the swarm, so a dynamic
      position-based topology is supported), but the cost/benefit is clearly
      negative: Delaunay is only practical in 2D/3D — the pure-Rust crates
      (`delaunator`, `spade`) are 2D-only and an n-D triangulation needs
      `qhull` (C++ FFI), which the core forbids and which explodes
      combinatorially, so the topology would be unusable for the higher-
      dimensional problems PSO targets. It would also require recomputing and
      caching the triangulation each iteration via interior mutability (the
      `neighbors(&self, …)` signature is immutable), and in practice it rarely
      beats ring or Von Neumann. The global/ring/Von Neumann/random set already
      covers the standard topology spectrum.
- [x] NumPy interop for `vectorized=True` (via the `numpy` crate): the swarm is
      passed as a contiguous NumPy array and results read back as a slice.
      Matches `pyswarms` on expensive objectives; cheap objectives still favor a
      fully-NumPy library (closing that fully would need a flat-matrix swarm).

## ✅ Phase 3 — Python layer (DONE)
- [x] `pyo3` active in `crates/pso-py/Cargo.toml`.
- [x] `crates/pso-py` added to the workspace.
- [x] `minimize` implemented: accepts a Python callable OR a native benchmark.
- [x] Real and integer space from Python.
- [x] `PsoResult` exposed with convergence and history.
- [x] Python error propagation.
- [x] Polished API in `python/turboswarm/__init__.py` (`import turboswarm`).
- [x] Expose Phase 2 variants/topologies by name (`constriction`,
      `ring`, `vonneumann`) + new native benchmarks.
- [~] Publish to TestPyPI — **WON'T DO** (superseded). TestPyPI was a dry-run
      step before the first public release; `turboswarm` is now published
      directly to real PyPI (and `turboswarm-core` to crates.io) via the tagged
      release workflows, so a TestPyPI upload no longer adds value.

## ✅ Phase 4 — Visualization and notebooks (DONE)
- [x] `viz.plot_convergence` with matplotlib.
- [x] `viz.animate_swarm` (contour + animated swarm, 2D).
- [x] `viz.compare` (overlay curves of several variants).
- [x] `viz.plot_pareto` (objective space of a multi-objective front).
- [x] Example notebooks: quickstart, variants & topologies, integer/mixed/
      constraints, multi-objective (`notebooks/`).

## Future ideas
- [x] CEC-family benchmark functions (`bent_cigar`, `discus`, `elliptic`,
      `zakharov`, `levy`, `expanded_schaffer`): canonical base functions of the
      CEC suites, native + Python mirror (official shift/rotation data not bundled).
- [x] Hyperparameter sensitivity analysis (`turboswarm.sweep`): Cartesian
      product over hyperparameters + seed aggregation; `viz.plot_sensitivity`.
- [x] Parallelization with `rayon` (`minimize_parallel`).
- [x] Constraints (penalty).
- [x] Stopping criteria (stagnation, tolerance, target, eval/time budget).
- [x] Multi-objective (MOPSO) as a separate module (`turboswarm_core::mopso`).
- [x] MOPSO turbulence/mutation operator (`mutation_rate`).
- [x] MOPSO hypervolume metric (`mopso::hypervolume`, WFG algorithm;
      `ParetoFront.hypervolume` in Python).
- [x] MOPSO grid-based archive (Coello's adaptive hypercube grid):
      `grid_divisions=` in `minimize_multi` / `MopsoParams.grid_divisions`.
- [x] Equality constraints (`equality_constraints=`, quadratic penalty) and a
      `repair=` operator in the Python `minimize`.

## Backlog — candidate improvements (not scheduled)

Unprioritized ideas, grouped by the project's priorities (viz > comparison >
clarity > performance). Pull into a phase when picked up.

### Visualization (priority #1)
- [ ] **WASM build** of the core (`wasm-bindgen`) → interactive swarm animations
      in the browser / on the web. Strong differentiator vs `pyswarms`.
- [ ] **Interactive backend (Plotly)** alongside matplotlib: zoom, per-particle
      hover, iteration slider.
- [ ] **3D / projected animation** for >2D problems (PCA or dimension pairs),
      lifting the current 2D-only limit of `animate_swarm`.
- [ ] **History export** to CSV/Parquet for downstream analysis.

### Algorithm comparison (priority #2)
- [ ] **Benchmarking suite with statistical tests**: multi-seed runner +
      Wilcoxon / Friedman + ranking tables.
- [ ] **Official CEC shift/rotation data** loader (complete CEC functions, not
      just the base forms already added).
- [ ] **Performance profiles / aggregated convergence plots** across functions.
- [ ] **More Pareto-front metrics**: IGD, spacing, spread (today only
      hypervolume).

### Variants & operators (extensibility = core's reason to exist)
- [ ] **Classic variants**, one `Velocity` impl each: Bare-bones PSO (Kennedy),
      QPSO (quantum-behaved), CLPSO (comprehensive learning), SPSO-2011
      (rotation-invariant), APSO (adaptive / self-tuning).
- [ ] **GCPSO** (guaranteed convergence) + stagnation-triggered restart.
- [ ] **Quasi-random initialization** (Sobol / LHS) as an alternative to uniform.
- [ ] **Niching / multimodal** (speciation or lbest-ring niching).
- [ ] **Alternative constraint handling**: Deb's feasibility rules or
      ε-constraint (today only penalty/repair).
- [ ] **Advanced MOPSO**: SMPSO (speed-constrained) or MOEA/D-style decomposition.

### Clarity / DX (priority #3)
- [ ] **Type stubs (`.pyi`)** for the native module (autocompletion + mypy).
- [ ] **Checkpoint / serialization** of optimizer state (resume long runs).
- [ ] **Documentation site** (mkdocs + docs.rs) with the "implement a trait =
      new variant" guide.

### Performance (priority #4)
- [ ] **Flat contiguous swarm matrix** (1D `Vec<f64>` + strides) instead of
      `Vec<Vec<f64>>` → better cache locality; closes the NumPy gap on cheap
      objectives.
- [ ] **GPU acceleration** (wgpu) for massive batch objective evaluation.
- [ ] **Cooperative coevolution (CCPSO2)** for high-dimensional problems.
