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

## 🟢 Phase 5 — Reaching (and passing) feature parity with other libraries
- [x] Inequality constraints (penalty method) in the Python `minimize`
      (`constraints=`, `penalty=`).
- [x] Parallel objective evaluation from Rust (`minimize_parallel`, `rayon`).
- [x] Binary optimization helper (`binary=True` in the Python `minimize`).
- [ ] `Pyramid` topology (Delaunay-based). `pyswarms` has this. Deliberately
      deferred: it needs a computational-geometry dependency for a niche
      topology that rarely beats Von Neumann or ring.
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
- [ ] Publish to TestPyPI.

## ✅ Phase 4 — Visualization and notebooks (DONE)
- [x] `viz.plot_convergence` with matplotlib.
- [x] `viz.animate_swarm` (contour + animated swarm, 2D).
- [x] `viz.compare` (overlay curves of several variants).
- [x] `viz.plot_pareto` (objective space of a multi-objective front).
- [x] Example notebooks: quickstart, variants & topologies, integer/mixed/
      constraints, multi-objective (`notebooks/`).

## Future ideas
- [x] Hyperparameter sensitivity analysis (`turboswarm.sweep`): Cartesian
      product over hyperparameters + seed aggregation; `viz.plot_sensitivity`.
- [x] Parallelization with `rayon` (`minimize_parallel`).
- [x] Constraints (penalty).
- [x] Stopping criteria (stagnation, tolerance, target, eval/time budget).
- [x] Multi-objective (MOPSO) as a separate module (`turboswarm_core::mopso`).
- [x] MOPSO turbulence/mutation operator (`mutation_rate`).
- [x] MOPSO hypervolume metric (`mopso::hypervolume`, WFG algorithm;
      `ParetoFront.hypervolume` in Python).
- [ ] MOPSO refinements: grid-based archive (Coello's).
- [x] Equality constraints (`equality_constraints=`, quadratic penalty) and a
      `repair=` operator in the Python `minimize`.
