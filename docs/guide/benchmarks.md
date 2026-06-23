# Benchmarks

Standard test functions with known global optima, useful to validate the
optimizer and measure real error. They are implemented natively in Rust (fast,
no GIL) and mirrored in pure Python (`turboswarm.benchmarks`) for plotting.

| Name | Recommended bound (±) | Optimum | Notes |
|------|----------------------|---------|-------|
| `sphere` | 5.12 | f(0) = 0 | Convex, unimodal |
| `rastrigin` | 5.12 | f(0) = 0 | Highly multimodal |
| `rosenbrock` | 2.048 | f(1,…,1) = 0 | Narrow banana valley |
| `ackley` | 32.768 | f(0) = 0 | Nearly flat far out, narrow well |
| `griewank` | 600 | f(0) = 0 | Many regular local minima |
| `schwefel` | 500 | f(420.97,…) = 0 | **Optimum far from the origin** |

!!! tip "Example"
    Schwefel's optimum sits near 420.97 per dimension, not at the origin — a
    good way to show that centering the search at 0 can be misleading.

## Using a benchmark by name

```python
r = pso.minimize("ackley", bounds=[(-32.768, 32.768)] * 2, seed=1)
```

## Metadata helper

`benchmark_info(name)` returns `(bound, optimum)`, so you can size the domain
without hardcoding it:

```python
bound, optimum = pso.benchmark_info("schwefel")
bounds = [(-bound, bound)] * 2
r = pso.minimize("schwefel", bounds=bounds, n_particles=60, max_iter=400, seed=3)
print(r.best_value - optimum)   # gap to the known optimum
```

## Pure-Python mirrors

For plotting (e.g. the contour map in [Visualization](visualization.md)) use
the Python versions, which take a list and return a float:

```python
pso.benchmarks.rastrigin([0.0, 0.0])   # -> 0.0
pso.benchmarks.BOUNDS["rastrigin"]      # -> 5.12
```
