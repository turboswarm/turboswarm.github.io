"""Fair benchmark of turboswarm against popular Python PSO libraries.

Compares wall-clock time and solution quality on standard functions, using the
SAME swarm size, iterations and coefficients for every library. Each library is
used idiomatically:

  - turboswarm (native): objective runs in Rust (its intended fast path).
  - turboswarm (py):      same Python callable as the competitors.
  - pyswarms:             vectorized NumPy objective (how it is meant to be used).
  - pyswarm:              scalar Python objective.

turboswarm runs with record_history=False so it does the same work as the
others (no trajectory recording).

Usage:
    pip install pyswarms pyswarm
    python scripts/bench_vs_libs.py
    python scripts/bench_vs_libs.py --markdown   # emit a Markdown table

Numbers are machine-dependent; always reproduce on your own hardware.
"""
import argparse
import statistics
import time

import numpy as np

import turboswarm as ts

# ---- Shared configuration (identical for every library) --------------------
DIM = 10
N_PARTICLES = 40
ITERS = 200
REPEATS = 5
W, C1, C2 = 0.729, 1.49445, 1.49445

FUNCS = {
    "sphere": 5.12,
    "rastrigin": 5.12,
    "ackley": 32.768,
}


# ---- Scalar objectives (for pyswarm and the turboswarm Python callable) -----
def sphere(x):
    return float(np.sum(np.square(x)))


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    x = np.asarray(x)
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


SCALAR = {"sphere": sphere, "rastrigin": rastrigin, "ackley": ackley}


# ---- Vectorized objectives (idiomatic pyswarms usage) ----------------------
def sphere_v(X):
    return np.sum(X * X, axis=1)


def rastrigin_v(X):
    return 10 * X.shape[1] + np.sum(X * X - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley_v(X):
    n = X.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(X * X, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / n) + 20 + np.e)


VECTOR = {"sphere": sphere_v, "rastrigin": rastrigin_v, "ackley": ackley_v}


def timed(fn):
    """Run fn once to warm up, then REPEATS times; return (median_ms, best)."""
    fn()  # warm-up (import/JIT/allocation effects)
    times, values = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        best = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
        values.append(best)
    return statistics.median(times), statistics.median(values)


def run_turboswarm_native(name, bound):
    def go():
        r = ts.minimize(name, bounds=[(-bound, bound)] * DIM,
                        n_particles=N_PARTICLES, max_iter=ITERS,
                        w=W, c1=C1, c2=C2, seed=0, record_history=False)
        return r.best_value
    return timed(go)


def run_turboswarm_py(name, bound):
    f = SCALAR[name]
    def go():
        r = ts.minimize(f, bounds=[(-bound, bound)] * DIM,
                        n_particles=N_PARTICLES, max_iter=ITERS,
                        w=W, c1=C1, c2=C2, seed=0, record_history=False)
        return r.best_value
    return timed(go)


def run_pyswarms(name, bound):
    from pyswarms.single import GlobalBestPSO
    f = VECTOR[name]
    bounds = (np.full(DIM, -bound), np.full(DIM, bound))
    opts = {"c1": C1, "c2": C2, "w": W}
    def go():
        np.random.seed(0)
        opt = GlobalBestPSO(n_particles=N_PARTICLES, dimensions=DIM,
                            options=opts, bounds=bounds)
        cost, _ = opt.optimize(f, iters=ITERS, verbose=False)
        return cost
    return timed(go)


def run_pyswarm(name, bound):
    from pyswarm import pso
    f = SCALAR[name]
    lb, ub = [-bound] * DIM, [bound] * DIM
    def go():
        np.random.seed(0)
        res = pso(f, lb, ub, swarmsize=N_PARTICLES, maxiter=ITERS,
                  omega=W, phip=C1, phig=C2)
        # This pyswarm build returns a scipy-like OptimizeResult.
        return float(res["fun"]) if hasattr(res, "keys") else float(res[1])
    return timed(go)


RUNNERS = [
    ("turboswarm (native)", run_turboswarm_native),
    ("turboswarm (py)", run_turboswarm_py),
    ("pyswarms", run_pyswarms),
    ("pyswarm", run_pyswarm),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true",
                        help="print a Markdown table instead of plain text")
    args = parser.parse_args()

    print(f"# dim={DIM}, particles={N_PARTICLES}, iters={ITERS}, "
          f"repeats={REPEATS} (median)\n")

    rows = []
    for name, bound in FUNCS.items():
        for label, runner in RUNNERS:
            ms, best = runner(name, bound)
            rows.append((name, label, ms, best))
            print(f"{name:10s} {label:22s} {ms:9.2f} ms   best={best:.3e}")
        print()

    if args.markdown:
        print("\n| Function | Library | Time (ms) | Best value |")
        print("|----------|---------|----------:|-----------:|")
        for name, label, ms, best in rows:
            print(f"| {name} | {label} | {ms:.2f} | {best:.2e} |")


if __name__ == "__main__":
    main()
