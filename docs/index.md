# turboswarm

**Particle Swarm Optimization** with a compute core in **Rust** and an API in
**Python**. Extensible, supports **real, integer and mixed** variables, with
visualization first, easy algorithm comparison, and readable code.

## Highlights

- **Velocity variants:** inertia (Shi-Eberhart), constriction (Clerc-Kennedy)
  and FIPS (Fully Informed PSO).
- **Topologies:** global (gbest), ring (lbest), Von Neumann (2D grid) and random.
- **Spaces:** continuous, integer, binary and mixed (per-dimension type).
- **Multi-objective (MOPSO):** Pareto front via archive + crowding distance.
- **Constraints** (penalty), **run control** (target / budgets / stagnation /
  callback) and **boundary handling** (clamp, reflect, wrap, reinit).
- **Performance:** velocity clamp, parallel evaluation (`rayon`) and a
  vectorized objective path.
- **Benchmarks:** sphere, rastrigin, rosenbrock, ackley, griewank and schwefel,
  each with metadata (recommended bound and known optimum).
- **Visualization** (Python): convergence curves, variant comparison, a 2D
  animated swarm and Pareto-front plots.
- **Reproducibility:** every experiment accepts a `seed`.

## The core idea

The PSO loop knows nothing about any concrete variant. Everything that changes
between variants lives behind three traits — `SearchSpace`, `Velocity` and
`Topology`. Adding a new variant means implementing one trait, without touching
the core. See [Architecture](architecture.md) and [Extending](extending.md).

## Two ways to use it

=== "Python"

    ```python
    import turboswarm as pso

    # Same (min, max) on every axis -> pass the pair once with `dim`.
    dim = 2
    search_domain = (-5.12, 5.12)

    result = pso.minimize(
        "rastrigin",
        bounds=search_domain,
        dim=dim,
        velocity="fips",
        topology="ring",
        seed=1,
    )
    print(result.best_position, result.best_value)
    ```

=== "Rust"

    ```rust
    use turboswarm_core::prelude::*;
    use turboswarm_core::benchmarks::rastrigin;

    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let velocity = InertiaVelocity::new(0.729, 1.49445, 1.49445);
    let params = PsoParams { seed: Some(42), ..Default::default() };

    let result = Pso::new(space, velocity, GlobalBest::new(), params)
        .minimize(rastrigin);
    ```

Start with [Installation](installation.md) and the
[Getting started](getting-started.md) tutorial.
