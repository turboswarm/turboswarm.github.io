"""turboswarm — Particle Swarm Optimization with a Rust core, Python API.

The computation runs in Rust (native module `turboswarm_native`); this Python
layer offers a convenient API and visualization utilities.

Example:
    >>> import turboswarm as pso
    >>> r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)]*2, seed=42)
    >>> r.best_value < 1e-3
    True

    >>> # your own function (Python)
    >>> r = pso.minimize(lambda x: sum(xi**2 for xi in x), bounds=[(-5,5)]*3)

    >>> # integer variables
    >>> r = pso.minimize(f, bounds=[(-10,10)]*2, integer=True)

Variants (`velocity=`): "inertia", "constriction", "fips".
Topologies (`topology=`): "global", "ring", "vonneumann".
(FIPS performs better with local topologies: "ring" or "vonneumann".)
Native benchmarks: "sphere", "rastrigin", "rosenbrock", "ackley",
"griewank", "schwefel", plus the CEC-family functions "bent_cigar", "discus",
"elliptic", "zakharov", "levy", "expanded_schaffer".

Hyperparameter sensitivity (`sweep`): run PSO over a Cartesian product of
hyperparameter values and aggregate over seeds; visualize with
`viz.plot_sensitivity`.
"""

import logging

from .turboswarm_native import (
    minimize,
    minimize_multi,
    minimize_grey,
    hypervolume,
    PsoResult,
    ParetoFront,
    GreyResult,
    benchmark_info,
    grey_benchmark_info,
)
from .sensitivity import sweep, SweepResult
from . import benchmarks, viz, integrations

# Library best practice: attach a NullHandler so importing turboswarm never emits
# log output on its own. Applications opt in by configuring logging themselves
# (e.g. logging.basicConfig(level=logging.INFO)).
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.4.0"
__all__ = [
    "minimize",
    "minimize_multi",
    "minimize_grey",
    "hypervolume",
    "sweep",
    "SweepResult",
    "PsoResult",
    "ParetoFront",
    "GreyResult",
    "benchmark_info",
    "grey_benchmark_info",
    "benchmarks",
    "viz",
    "integrations",
    "__version__",
]
