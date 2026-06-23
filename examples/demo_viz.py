"""Visual demo of turboswarm (no notebook needed).

Run after `maturin develop`:
    python examples/demo_viz.py

Opens two windows:
  1. Convergence comparison of three variants on Rastrigin.
  2. Animated 2D swarm over the Rastrigin contour map.
"""
import logging

import matplotlib.pyplot as plt

import turboswarm as pso

# Configure logging at the application level. This also surfaces the INFO logs
# emitted by turboswarm.viz (run comparison, animation frame counts).
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("demo_viz")

BENCH = "rastrigin"
bound, _ = pso.benchmark_info(BENCH)
bounds = [(-bound, bound)] * 2

# 1) Compare variants on the same problem and seed.
runs = {
    "inertia / global": pso.minimize(BENCH, bounds=bounds, velocity="inertia",
                                      topology="global", seed=42),
    "constriction / ring": pso.minimize(BENCH, bounds=bounds, velocity="constriction",
                                         topology="ring", seed=42),
    "fips / vonneumann": pso.minimize(BENCH, bounds=bounds, velocity="fips",
                                      topology="vonneumann", seed=42),
}
for name, r in runs.items():
    log.info("%-24s -> best_value = %.3e", name, r.best_value)

pso.viz.compare(runs)
plt.tight_layout()

# 2) Animate the swarm of the FIPS run over the contour map.
best_run = runs["fips / vonneumann"]
anim = pso.viz.animate_swarm(best_run, pso.benchmarks.rastrigin, bounds)

plt.show()  # close the convergence window to see the animation
