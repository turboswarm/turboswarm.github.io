# Hyperparameter sensitivity

PSO has several hyperparameters (`w`, `c1`, `c2`, swarm size, topology, …) and
their best values are problem-dependent. `turboswarm.sweep` runs the optimizer
over the **Cartesian product** of the value lists you give it, so you can see
which hyperparameters actually move the solution quality.

Because PSO is stochastic, each combination can be repeated over several
**seeds** and aggregated (mean / std / min / max of the best value).

## Example

```python
import turboswarm as pso

sweep = pso.sweep(
    "rastrigin", bounds=(-5.12, 5.12), dim=2,
    grid={"w": [0.4, 0.7, 0.9], "c1": [1.0, 2.0]},   # 3 x 2 = 6 combinations
    seeds=5,                                          # repeat each one 5 times
    n_particles=30, max_iter=100,                     # fixed for every run
)

print(sweep.best())          # combination with the lowest mean best value
for rec in sweep:
    print(rec["w"], rec["c1"], rec["mean"], rec["std"])
```

Each record is a `dict` with the swept hyperparameters plus `mean`, `std`
(population), `min`, `max`, the raw `values` list and `n` (number of seeds).

## Result object

`sweep` returns a `SweepResult` that is iterable and indexable over its records:

| Member | Meaning |
|--------|---------|
| `len(result)` | number of combinations |
| `result.best(metric="mean")` | record with the lowest value of `metric` |
| `result.to_dataframe()` | the records as a pandas `DataFrame` (pandas imported lazily — not a hard dependency) |

## What you can sweep

The keys of `grid` are any keyword accepted by [`minimize`](../api/python.md) —
e.g. `w`, `c1`, `c2`, `n_particles`, `max_iter`, `velocity`, `topology`,
`v_max`. Anything you do **not** sweep is passed as a fixed keyword argument and
forwarded to every run. A key cannot be both swept and fixed, and repetitions go
through `seeds=` (do not pass a fixed `seed`). History recording is turned off by
default for speed.

## Visualizing

`viz.plot_sensitivity` draws a line for one swept hyperparameter (with error
bars from the spread) or a heatmap for two:

```python
import matplotlib.pyplot as plt

pso.viz.plot_sensitivity(sweep, x="w")            # 1D line
pso.viz.plot_sensitivity(sweep, x="w", y="c1")    # 2D heatmap
plt.show()
```

When the sweep varies more hyperparameters than the ones plotted, the points are
marginalized (averaged) over the rest.
