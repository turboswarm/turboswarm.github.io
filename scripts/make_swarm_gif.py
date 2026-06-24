#!/usr/bin/env python
"""Render the README's swarm animation to an animated GIF.

Runs a small 2-D PSO on Rastrigin (recording history) and saves the contour +
swarm animation with matplotlib's Pillow writer. Re-run after changing the demo
to refresh the GIF:

    python scripts/make_swarm_gif.py            # -> docs/assets/swarm.gif
    python scripts/make_swarm_gif.py out.gif    # custom path
"""
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
from matplotlib.animation import PillowWriter  # noqa: E402

import turboswarm as ts  # noqa: E402
from turboswarm import benchmarks as bench  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/swarm.gif"
BOUND = bench.BOUNDS["rastrigin"]


def main():
    result = ts.minimize(
        "rastrigin",
        bounds=(-BOUND, BOUND),
        dim=2,
        n_particles=30,
        max_iter=50,
        seed=7,
        record_history=True,
    )
    anim = ts.viz.animate_swarm(
        result, bench.rastrigin, [(-BOUND, BOUND), (-BOUND, BOUND)], interval=120
    )
    anim.save(OUT, writer=PillowWriter(fps=10), dpi=80)
    print(f"wrote {OUT} ({len(result.history)} frames)")


if __name__ == "__main__":
    main()
