"""Visualization helpers. Consume the `PsoResult` from the Rust core.

Requires matplotlib. Functions:
  - plot_convergence(result, label=None, ax=None)
  - animate_swarm(result, function, bounds)   # 2D swarm over a contour
  - compare(results)                          # overlay convergence curves
  - plot_pareto(front, ax=None)               # objective space of a ParetoFront
  - plot_sensitivity(sweep, x, y=None)        # hyperparameter sweep (line/heatmap)
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


def _ordered_unique(values):
    """Unique values preserving first-seen order (matches the grid order)."""
    return list(dict.fromkeys(values))


def plot_sensitivity(sweep, x, y=None, metric="mean", ax=None):
    """Visualize a hyperparameter sweep (the result of ``turboswarm.sweep``).

    1D (``y=None``): a line of ``metric`` vs the ``x`` hyperparameter. When the
    sweep varies other hyperparameters too, the points are marginalized over
    them (mean across the group) and the spread is shown as error bars.

    2D (``y`` given): a heatmap of ``metric`` over the ``x`` (columns) and ``y``
    (rows) grid, again marginalizing over any other swept hyperparameters.

    Args:
        sweep (SweepResult): from ``turboswarm.sweep``.
        x (str): hyperparameter for the x-axis (a key in ``sweep.param_names``).
        y (str | None): hyperparameter for the y-axis -> heatmap; ``None`` -> line.
        metric (str): which per-combination statistic to plot (``"mean"``,
            ``"min"``, ``"max"``, ``"std"``). Defaults to ``"mean"``.
        ax: optional matplotlib Axes.

    Returns:
        The matplotlib Axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    records = list(sweep)
    if not records:
        raise ValueError("empty sweep")
    for name in (x, y) if y is not None else (x,):
        if name not in records[0]:
            raise ValueError(f"{name!r} is not a swept hyperparameter")

    if ax is None:
        _, ax = plt.subplots()

    if y is None:
        xs = _ordered_unique(r[x] for r in records)
        means, errs = [], []
        for xv in xs:
            vals = [r[metric] for r in records if r[x] == xv]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        positions = range(len(xs))
        ax.errorbar(positions, means, yerr=errs, marker="o", capsize=4)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel(x)
        ax.set_ylabel(f"{metric} best value")
        ax.set_title(f"Sensitivity to {x}")
        logger.info("plotting 1D sensitivity over %r (%d levels)", x, len(xs))
        return ax

    xs = _ordered_unique(r[x] for r in records)
    ys = _ordered_unique(r[y] for r in records)
    grid = np.full((len(ys), len(xs)), np.nan)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            cell = [r[metric] for r in records if r[x] == xv and r[y] == yv]
            if cell:
                grid[j, i] = float(np.mean(cell))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(v) for v in ys])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Sensitivity: {metric} best value")
    ax.figure.colorbar(im, ax=ax, label=f"{metric} best value")
    logger.info("plotting 2D sensitivity heatmap %r x %r (%dx%d)",
                x, y, len(xs), len(ys))
    return ax
