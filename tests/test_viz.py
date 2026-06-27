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


# --- 3D projection for >2D problems (issue #2) ------------------------------
def test_projected_animation_pca_and_dims():
    from matplotlib.animation import FuncAnimation

    r = pso.minimize("sphere", bounds=[(-5, 5)] * 5, seed=2, n_particles=10,
                     max_iter=8)
    anim = pso.viz.animate_swarm_projected(r, function=pso.benchmarks.sphere)
    assert isinstance(anim, FuncAnimation)
    anim2 = pso.viz.animate_swarm_projected(r, dims=(0, 1, 2))
    assert isinstance(anim2, FuncAnimation)


def test_projected_animation_rejects_low_dim():
    r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1, max_iter=5)
    with pytest.raises(ValueError):
        pso.viz.animate_swarm_projected(r)


# --- Interactive Plotly backend (issue #1) ----------------------------------
def test_plotly_convergence_and_compare():
    pytest.importorskip("plotly")
    r = _run()
    fig = pso.viz.plotly_convergence(r)
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.data) == 1
    fig2 = pso.viz.plotly_compare({"a": r, "b": _run()})
    assert len(fig2.data) == 2


def test_plotly_pareto():
    pytest.importorskip("plotly")
    front = pso.minimize_multi(
        lambda x: [sum(v * v for v in x), sum((v - 2) ** 2 for v in x)],
        bounds=[(-5, 5)] * 2, seed=1, max_iter=20)
    fig = pso.viz.plotly_pareto(front)
    assert len(fig.data[0].x) == len(front)
