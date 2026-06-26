"""Tests for the visualization helpers (run with pytest)."""
import pytest

import turboswarm as pso

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless backend for tests


def _run():
    return pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=3,
                        n_particles=12, max_iter=8)


def test_plot_surface_returns_3d_axes():
    ax = pso.viz.plot_surface(pso.benchmarks.rastrigin, [(-5.12, 5.12)] * 2)
    assert ax.name == "3d"


def test_plot_surface_with_points():
    r = _run()
    ax = pso.viz.plot_surface(pso.benchmarks.rastrigin, [(-5.12, 5.12)] * 2,
                              points=r.history[-1])
    assert ax.name == "3d"


def test_animate_swarm_3d_returns_animation():
    from matplotlib.animation import FuncAnimation

    r = _run()
    anim = pso.viz.animate_swarm_3d(r, pso.benchmarks.rastrigin,
                                    [(-5.12, 5.12)] * 2)
    assert isinstance(anim, FuncAnimation)


def test_3d_helpers_reject_non_2d():
    r = pso.minimize("sphere", bounds=[(-5, 5)] * 3, seed=1, max_iter=5)
    with pytest.raises(ValueError):
        pso.viz.plot_surface(pso.benchmarks.sphere, [(-5, 5)] * 3)
    with pytest.raises(ValueError):
        pso.viz.animate_swarm_3d(r, pso.benchmarks.sphere, [(-5, 5)] * 3)


def test_animate_swarm_3d_requires_history():
    r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1, max_iter=5,
                     record_history=False)
    with pytest.raises(ValueError):
        pso.viz.animate_swarm_3d(r, pso.benchmarks.sphere, [(-5, 5)] * 2)
