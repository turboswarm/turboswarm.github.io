"""Reproducible benchmark suite: turboswarm vs popular Python PSO libraries.

Compares wall-clock time and solution quality across standard functions and
several dimensions, using the SAME swarm size, iteration budget and PSO
coefficients for every library. Each library is driven idiomatically:

  - turboswarm (native): objective runs in Rust (its intended fast path).
  - turboswarm (py):      same Python callable as the competitors.
  - pyswarms:             vectorized NumPy objective.
  - pyswarm:              scalar Python objective.
  - pymoo:                vectorized Problem, fixed-coefficient PSO.

Methodology
-----------
Identical configuration for all libraries (see CONFIG below). For each
(library, function, dimension) we run one optimization per seed, after a
warm-up call, and report the MEDIAN time and MEDIAN best objective across
seeds. turboswarm runs with record_history=False so it does the same work as
the others.

Outputs (under benches/results/):
  - results.csv   : one row per (function, dim, library) with time and quality.
  - meta.json     : machine + library versions for provenance.
  - speedup.png   : speedup of turboswarm (native) vs pyswarms across dimensions.

Usage:
    pip install pyswarms pyswarm pymoo
    python benches/bench_suite.py
    python benches/bench_suite.py --markdown   # also print a Markdown table

Absolute numbers are machine-dependent; always reproduce on your own hardware.
"""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

import turboswarm as ts

# ---- Shared configuration (identical for every library) --------------------
N_PARTICLES = 40
ITERS = 200
SEEDS = [0, 1, 2, 3, 4]
DIMS = [2, 10, 30]
W, C1, C2 = 0.729, 1.49445, 1.49445

# function name -> symmetric bound used on every dimension
FUNCS = {
    "sphere": 5.12,
    "rastrigin": 5.12,
    "ackley": 32.768,
    "rosenbrock": 5.12,
    "griewank": 600.0,
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---- Scalar objectives (pyswarm and the turboswarm Python callable) --------
def sphere(x):
    x = np.asarray(x)
    return float(np.sum(x * x))


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    x = np.asarray(x)
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def griewank(x):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))))


SCALAR = {"sphere": sphere, "rastrigin": rastrigin, "ackley": ackley,
          "rosenbrock": rosenbrock, "griewank": griewank}


# ---- Vectorized objectives (idiomatic pyswarms / pymoo usage) --------------
def sphere_v(X):
    return np.sum(X * X, axis=1)


def rastrigin_v(X):
    return 10 * X.shape[1] + np.sum(X * X - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley_v(X):
    n = X.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(X * X, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / n) + 20 + np.e)


def rosenbrock_v(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
                  + (1 - X[:, :-1]) ** 2, axis=1)


def griewank_v(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1 + np.sum(X * X, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


VECTOR = {"sphere": sphere_v, "rastrigin": rastrigin_v, "ackley": ackley_v,
          "rosenbrock": rosenbrock_v, "griewank": griewank_v}


def timed_over_seeds(run_one):
    """run_one(seed) -> best. Warm up once, then run for every seed.

    Returns (median_ms, median_best)."""
    run_one(SEEDS[0])  # warm-up (import/allocation/JIT effects)
    times, values = [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        best = run_one(seed)
        times.append((time.perf_counter() - t0) * 1000.0)
        values.append(best)
    return statistics.median(times), statistics.median(values)


# ---- Per-library runners ---------------------------------------------------
def run_turboswarm_native(name, bound, dim):
    def one(seed):
        r = ts.minimize(name, bounds=[(-bound, bound)] * dim,
                        n_particles=N_PARTICLES, max_iter=ITERS,
                        w=W, c1=C1, c2=C2, seed=seed, record_history=False)
        return r.best_value
    return timed_over_seeds(one)


def run_turboswarm_py(name, bound, dim):
    f = SCALAR[name]
    def one(seed):
        r = ts.minimize(f, bounds=[(-bound, bound)] * dim,
                        n_particles=N_PARTICLES, max_iter=ITERS,
                        w=W, c1=C1, c2=C2, seed=seed, record_history=False)
        return r.best_value
    return timed_over_seeds(one)


def run_pyswarms(name, bound, dim):
    from pyswarms.single import GlobalBestPSO
    f = VECTOR[name]
    bounds = (np.full(dim, -bound), np.full(dim, bound))
    opts = {"c1": C1, "c2": C2, "w": W}
    def one(seed):
        np.random.seed(seed)
        opt = GlobalBestPSO(n_particles=N_PARTICLES, dimensions=dim,
                            options=opts, bounds=bounds)
        cost, _ = opt.optimize(f, iters=ITERS, verbose=False)
        return cost
    return timed_over_seeds(one)


def run_pyswarm(name, bound, dim):
    from pyswarm import pso
    f = SCALAR[name]
    lb, ub = [-bound] * dim, [bound] * dim
    def one(seed):
        np.random.seed(seed)
        res = pso(f, lb, ub, swarmsize=N_PARTICLES, maxiter=ITERS,
                  omega=W, phip=C1, phig=C2)
        return float(res["fun"]) if hasattr(res, "keys") else float(res[1])
    return timed_over_seeds(one)


def run_pymoo(name, bound, dim):
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    fvec = VECTOR[name]

    class _Problem(Problem):
        def __init__(self):
            super().__init__(n_var=dim, n_obj=1, xl=-bound, xu=bound)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = fvec(X)

    def one(seed):
        # Fixed coefficients (adaptive=False) to match the other libraries.
        algorithm = PSO(pop_size=N_PARTICLES, w=W, c1=C1, c2=C2, adaptive=False)
        res = pymoo_minimize(_Problem(), algorithm, ("n_gen", ITERS),
                             seed=seed, verbose=False)
        return float(res.F[0])
    return timed_over_seeds(one)


RUNNERS = [
    ("turboswarm (native)", run_turboswarm_native),
    ("turboswarm (py)", run_turboswarm_py),
    ("pyswarms", run_pyswarms),
    ("pyswarm", run_pyswarm),
    ("pymoo", run_pymoo),
]


def lib_versions():
    versions = {"turboswarm": getattr(ts, "__version__", "unknown")}
    for mod in ("pyswarms", "pyswarm", "pymoo", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    return versions


def make_speedup_figure(rows, path):
    """Speedup of turboswarm (native) vs pyswarms, per function across dims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # index: (func, dim, lib) -> time
    t = {(r["function"], r["dim"], r["library"]): r["time_ms"] for r in rows}
    funcs = list(FUNCS)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for fn in funcs:
        xs, ys = [], []
        for d in DIMS:
            native = t.get((fn, d, "turboswarm (native)"))
            comp = t.get((fn, d, "pyswarms"))
            if native and comp and native > 0:
                xs.append(d)
                ys.append(comp / native)
        if xs:
            ax.plot(xs, ys, marker="o", label=fn)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Speedup  (pyswarms time / turboswarm native time)")
    ax.set_title("turboswarm (native) speedup over pyswarms")
    ax.set_xticks(DIMS)
    ax.legend(title="function", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true",
                        help="also print a Markdown table to stdout")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"# particles={N_PARTICLES}, iters={ITERS}, seeds={SEEDS} "
          f"(median over seeds)\n")

    rows = []
    for name, bound in FUNCS.items():
        for dim in DIMS:
            for label, runner in RUNNERS:
                try:
                    ms, best = runner(name, bound, dim)
                except Exception as exc:  # keep going if one library fails
                    print(f"  ! {label} failed on {name}/dim={dim}: {exc}")
                    continue
                rows.append({"function": name, "dim": dim, "library": label,
                             "time_ms": round(ms, 3), "best": best})
                print(f"{name:10s} d={dim:<3d} {label:22s} "
                      f"{ms:9.2f} ms   best={best:.3e}")
        print()

    # ---- persist results + provenance --------------------------------------
    csv_path = RESULTS_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["function", "dim", "library", "time_ms", "best"])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "config": {"n_particles": N_PARTICLES, "iters": ITERS,
                   "seeds": SEEDS, "dims": DIMS, "w": W, "c1": C1, "c2": C2},
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor() or platform.machine(),
                    "python": platform.python_version()},
        "versions": lib_versions(),
    }
    (RESULTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    fig_path = RESULTS_DIR / "speedup.png"
    try:
        make_speedup_figure(rows, fig_path)
    except Exception as exc:
        print(f"  ! figure generation failed: {exc}")
        fig_path = None

    print(f"Wrote {csv_path}")
    print(f"Wrote {RESULTS_DIR / 'meta.json'}")
    if fig_path:
        print(f"Wrote {fig_path}")

    if args.markdown:
        print("\n| Function | Dim | Library | Time (ms) | Best value |")
        print("|----------|----:|---------|----------:|-----------:|")
        for r in rows:
            print(f"| {r['function']} | {r['dim']} | {r['library']} "
                  f"| {r['time_ms']:.2f} | {r['best']:.2e} |")


if __name__ == "__main__":
    main()
