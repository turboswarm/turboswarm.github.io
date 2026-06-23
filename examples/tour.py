"""A longer tour of turboswarm (Rust core, Python API).

Walks through the main features: native benchmarks, custom objectives,
integer optimization, variant and topology comparison, and reproducibility.

Visualization is OPTIONAL and off by default, so this script runs without
matplotlib. Enable it with flags:

    python examples/tour.py                 # text only (no matplotlib needed)
    python examples/tour.py --plot          # + convergence comparison plot
    python examples/tour.py --plot --animate # + animated 2D swarm
    python examples/tour.py -v              # verbose (DEBUG logging)

Run after building the module:
    maturin develop --release
"""
import argparse
import logging

import turboswarm as pso

log = logging.getLogger("tour")


def section(title):
    """Log a visual section header."""
    log.info("")
    log.info("=== %s ===", title)


def native_benchmarks():
    """Run every native benchmark on its recommended bounds."""
    section("Native benchmarks (computed in Rust)")
    for name in ["sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "schwefel"]:
        # benchmark_info gives the recommended symmetric bound and the optimum,
        # so we don't have to hardcode the domain of each function.
        bound, optimum = pso.benchmark_info(name)
        bounds = [(-bound, bound)] * 2
        r = pso.minimize(name, bounds=bounds, n_particles=40, max_iter=300, seed=42)
        gap = r.best_value - optimum
        log.info("%-10s bound=±%-7g best=%.3e (gap to optimum=%.3e)",
                 name, bound, r.best_value, gap)


def custom_objective():
    """Minimize a user-defined Python function."""
    section("Custom Python objective")
    # Shifted sphere: optimum at (2, 2, 2, 2), value 0.
    r = pso.minimize(
        lambda x: sum((xi - 2.0) ** 2 for xi in x),
        bounds=[(-10, 10)] * 4,
        seed=1,
    )
    log.info("argmin ≈ %s", [round(v, 4) for v in r.best_position])
    log.info("min    = %.3e", r.best_value)


def integer_problem():
    """Optimize over integer variables (discretization happens in decode)."""
    section("Integer optimization")
    # f(x) = (x0 - 3)^2 + (x1 + 2)^2, optimum at the integer point (3, -2).
    r = pso.minimize(
        lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2,
        bounds=[(-10, 10)] * 2,
        integer=True,
        seed=5,
    )
    log.info("integer argmin = %s (expected [3, -2])", r.best_position)
    log.info("value          = %g", r.best_value)


def compare_variants():
    """Compare velocity variants on a multimodal function. Returns the runs."""
    section("Variant comparison on Rastrigin")
    bound, _ = pso.benchmark_info("rastrigin")
    bounds = [(-bound, bound)] * 2
    runs = {}
    for velocity in ["inertia", "constriction", "fips"]:
        # FIPS shines with a local topology; the others use the global best.
        topology = "ring" if velocity == "fips" else "global"
        r = pso.minimize("rastrigin", bounds=bounds, velocity=velocity,
                         topology=topology, n_particles=40, max_iter=300, seed=7)
        runs[f"{velocity}/{topology}"] = r
        log.info("%-22s best=%.3e", f"{velocity}/{topology}", r.best_value)
    return runs


def compare_topologies():
    """Compare topologies with the same variant and seed."""
    section("Topology comparison on Ackley (inertia)")
    bound, _ = pso.benchmark_info("ackley")
    bounds = [(-bound, bound)] * 2
    for topology in ["global", "ring", "vonneumann"]:
        r = pso.minimize("ackley", bounds=bounds, topology=topology,
                         n_particles=49, max_iter=300, seed=3)
        log.info("%-12s best=%.3e", topology, r.best_value)


def reproducibility():
    """Same seed -> identical result; different seed -> (likely) different."""
    section("Reproducibility")
    a = pso.minimize("sphere", bounds=[(-5, 5)] * 3, seed=99)
    b = pso.minimize("sphere", bounds=[(-5, 5)] * 3, seed=99)
    c = pso.minimize("sphere", bounds=[(-5, 5)] * 3, seed=100)
    log.info("seed 99 == seed 99 : %s", a.best_position == b.best_position)
    log.info("seed 99 == seed 100: %s", a.best_position == c.best_position)


def visualize(runs, animate):
    """Optional matplotlib output. Imported lazily so the rest needs no deps."""
    section("Visualization")
    import matplotlib.pyplot as plt

    pso.viz.compare(runs)
    plt.tight_layout()

    anim = None
    if animate:
        # Animate the FIPS run over the Rastrigin contour map.
        best = next(iter(runs.values()))
        bound, _ = pso.benchmark_info("rastrigin")
        anim = pso.viz.animate_swarm(best, pso.benchmarks.rastrigin,
                                     [(-bound, bound)] * 2)

    log.info("showing plots (close the windows to finish)")
    plt.show()
    return anim  # keep a reference so the animation is not garbage-collected


def main():
    parser = argparse.ArgumentParser(description="turboswarm feature tour")
    parser.add_argument("--plot", action="store_true",
                        help="show the convergence comparison plot (needs matplotlib)")
    parser.add_argument("--animate", action="store_true",
                        help="also animate the 2D swarm (implies --plot)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    native_benchmarks()
    custom_objective()
    integer_problem()
    runs = compare_variants()
    compare_topologies()
    reproducibility()

    if args.plot or args.animate:
        visualize(runs, animate=args.animate)
    else:
        log.info("")
        log.info("(re-run with --plot or --animate to see the visualization)")


if __name__ == "__main__":
    main()
