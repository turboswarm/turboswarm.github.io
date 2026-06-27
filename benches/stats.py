"""Statistical comparison of the libraries benchmarked by ``bench_suite.py``.

Treats each (function, dimension) pair as a problem instance and the libraries
as the methods being compared (the standard Demsar 2006 setup). For both
solution quality (best value) and wall-clock time it reports:

  - a per-library mean rank (1 = best) across all instances,
  - the Friedman test (are the libraries different overall?),
  - pairwise Wilcoxon signed-rank tests of ``turboswarm (native)`` vs each other.

Lower is better for both metrics. Run after ``bench_suite.py``:

    python benches/stats.py
    python benches/stats.py --markdown
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

RESULTS = Path(__file__).resolve().parent / "results" / "results.csv"
REFERENCE = "turboswarm (native)"


def _load(path):
    rows = list(csv.DictReader(open(path)))
    libraries = list(dict.fromkeys(r["library"] for r in rows))
    instances = list(dict.fromkeys((r["function"], int(r["dim"])) for r in rows))
    quality, time = defaultdict(dict), defaultdict(dict)
    for r in rows:
        key = (r["function"], int(r["dim"]))
        quality[key][r["library"]] = float(r["best"])
        time[key][r["library"]] = float(r["time_ms"])
    return libraries, instances, quality, time


def _matrix(metric, libraries, instances):
    """rows = instances, cols = libraries."""
    return np.array([[metric[inst][lib] for lib in libraries]
                     for inst in instances], dtype=float)


def _mean_ranks(matrix):
    # rank each row ascending (1 = lowest = best), average ties.
    ranks = np.empty_like(matrix)
    for i, row in enumerate(matrix):
        order = row.argsort()
        rr = np.empty(len(row))
        rr[order] = np.arange(1, len(row) + 1)
        # average tied ranks
        _, inv, counts = np.unique(row, return_inverse=True, return_counts=True)
        avg = np.array([rr[row == v].mean() for v in row])
        ranks[i] = avg
    return ranks.mean(axis=0)


def analyze(metric_name, matrix, libraries):
    print(f"\n## {metric_name} (lower is better)\n")
    mean_ranks = _mean_ranks(matrix)
    order = mean_ranks.argsort()
    print(f"{'rank':>5}  {'library':<22}{'mean rank':>10}")
    for pos, j in enumerate(order, 1):
        print(f"{pos:>5}  {libraries[j]:<22}{mean_ranks[j]:>10.2f}")

    stat, p = friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    print(f"\nFriedman: chi2 = {stat:.3f}, p = {p:.2e} "
          f"({'libraries differ' if p < 0.05 else 'no significant difference'})")

    ref = libraries.index(REFERENCE)
    print(f"\nWilcoxon vs {REFERENCE} (pairwise):")
    for j, lib in enumerate(libraries):
        if j == ref:
            continue
        a, b = matrix[:, ref], matrix[:, j]
        if np.allclose(a, b):
            print(f"  {lib:<22} identical")
            continue
        try:
            _, pw = wilcoxon(a, b)
            faster = "turboswarm better" if np.median(a) < np.median(b) else "other better"
            print(f"  {lib:<22} p = {pw:.2e}  ({faster})")
        except ValueError as exc:
            print(f"  {lib:<22} n/a ({exc})")
    return mean_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true",
                    help="(reserved) print as Markdown")
    ap.parse_args()

    libraries, instances, quality, time = _load(RESULTS)
    print(f"# Statistical comparison over {len(instances)} problem instances "
          f"({len(libraries)} libraries)")
    analyze("Solution quality", _matrix(quality, libraries, instances), libraries)
    analyze("Wall-clock time", _matrix(time, libraries, instances), libraries)


if __name__ == "__main__":
    main()
