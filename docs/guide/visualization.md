# Visualization

Visualization is the project's first priority. The `turboswarm.viz` module consumes
a `PsoResult` and uses matplotlib. Because `record_history=True` by default,
every run is ready to animate.

!!! note
    `viz` imports matplotlib lazily, and the example scripts keep visualization
    optional (behind `--plot` / `--animate`), so the core runs without it.

## Convergence curve

```python
import matplotlib.pyplot as plt
import turboswarm as pso

r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=1)
pso.viz.plot_convergence(r)
plt.show()
```

## Comparing variants

`compare` takes a dict `{name: PsoResult}` and overlays their convergence
curves:

```python
runs = {
    "inertia/global": pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                                   velocity="inertia", topology="global", seed=7),
    "fips/ring": pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                              velocity="fips", topology="ring", seed=7),
}
pso.viz.compare(runs)
plt.show()
```

## Animating the swarm (2D)

```python
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=1)
anim = pso.viz.animate_swarm(r, pso.benchmarks.rastrigin, [(-5.12, 5.12)] * 2)
plt.show()
# In a notebook:  from IPython.display import HTML; HTML(anim.to_jshtml())
```

`animate_swarm` only supports 2D problems and requires `record_history=True`.

## Logging

`turboswarm` follows library conventions: it attaches a `NullHandler` and never
configures logging itself. Turn on log output from your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
# viz then logs run comparisons and animation frame counts on the
# "turboswarm.viz" logger.
```

## Ready-made demo

```bash
python examples/demo_viz.py        # compares variants, then animates
python examples/tour.py --animate  # full tour + optional animation
```
