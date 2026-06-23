"""Example of using turboswarm from Python (Rust core).

Run after compiling the module:
    maturin develop --release
    python examples/quickstart.py
"""
import logging

import turboswarm as pso

# The application (not the library) configures logging.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("quickstart")

# 1) Native benchmark (runs in Rust, no GIL — fast)
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=42)
log.info("Native Rastrigin: %s", r)

# 2) Your own function in Python
r = pso.minimize(
    lambda x: sum((xi - 2) ** 2 for xi in x),
    bounds=[(-10, 10)] * 4,
    seed=1,
)
log.info("Your own function: %s -> %g", r.best_position, r.best_value)

# 3) Integer variables
r = pso.minimize(
    lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2,
    bounds=[(-10, 10)] * 2,
    integer=True,
    seed=5,
)
log.info("Integer (optimum 3,-2): %s", r.best_position)

# 4) Visualization (requires matplotlib): see examples/demo_viz.py
