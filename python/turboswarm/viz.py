"""Visualization helpers. Consume the `PsoResult` from the Rust core.

Requires matplotlib. Functions:
  - plot_convergence(result, label=None, ax=None)
  - animate_swarm(result, function, bounds)   # 2D swarm over a contour
  - compare(results)                          # overlay convergence curves
  - plot_pareto(front, ax=None)               # objective space of a ParetoFront
"""
import logging

logger = logging.getLogger(__name__)


def plot_pareto(front, ax=None, labels=("objective 1", "objective 2")):
    """Scatter the objective values of a 2-objective ParetoFront."""
    import matplotlib.pyplot as plt

    objs = front.objectives
    if not objs or len(objs[0]) != 2:
        raise ValueError("plot_pareto supports exactly 2 objectives")
    logger.info("plotting Pareto front: %d solutions", len(objs))
    f1 = [o[0] for o in objs]
    f2 = [o[1] for o in objs]
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(f1, f2, s=20, c="tab:blue", edgecolors="white")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Pareto front")
    return ax


def plot_convergence(result, label=None, ax=None, log=True):
    """Draw the best-value-per-iteration curve."""
    import matplotlib.pyplot as plt

    logger.debug("plotting convergence: %d iterations, label=%r",
                 len(result.convergence), label)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.convergence, label=label)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value (log scale)" if log else "Best value")
    ax.set_title("PSO convergence")
    if label:
        ax.legend()
    return ax


def compare(results, log=True):
    """results: dict {name: PsoResult}. Overlays convergence curves."""
    import matplotlib.pyplot as plt

    logger.info("comparing %d runs: %s", len(results), ", ".join(results))
    _, ax = plt.subplots()
    for name, res in results.items():
        plot_convergence(res, label=name, ax=ax, log=log)
    return ax


def animate_swarm(result, function, bounds, interval=150):
    """Animate the 2D swarm over the contour map of `function`.

    `function`: callable that receives [x, y] and returns a float.
    `bounds`: [(xmin, xmax), (ymin, ymax)].
    Returns a FuncAnimation object (in a notebook: HTML(anim.to_jshtml())).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(bounds) != 2:
        raise ValueError("animate_swarm only supports 2D problems")
    if not result.history:
        raise ValueError("run minimize(..., record_history=True)")

    logger.info("building swarm animation: %d frames, %d particles",
                len(result.history), len(result.history[0]))
    (xmin, xmax), (ymin, ymax) = bounds
    gx = np.linspace(xmin, xmax, 200)
    gy = np.linspace(ymin, ymax, 200)
    gxx, gyy = np.meshgrid(gx, gy)
    gz = np.vectorize(lambda a, b: function([a, b]))(gxx, gyy)

    fig, ax = plt.subplots()
    ax.contourf(gxx, gyy, gz, levels=30, cmap="viridis")
    scat = ax.scatter([], [], c="red", s=20, edgecolors="white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame):
        pts = np.array(result.history[frame])  # [particle][dim]
        scat.set_offsets(pts)
        ax.set_title(f"Iteration {frame + 1}/{len(result.history)}")
        return (scat,)

    return FuncAnimation(
        fig, update, frames=len(result.history),
        init_func=init, interval=interval, blit=False
    )
