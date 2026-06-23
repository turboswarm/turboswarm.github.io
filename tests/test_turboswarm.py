"""Smoke/behaviour tests for the turboswarm Python API (run with pytest)."""
import numpy as np

import turboswarm as pso


def test_native_benchmark_converges():
    r = pso.minimize("sphere", bounds=(-5.12, 5.12), dim=2, seed=42)
    assert r.best_value < 1e-3
    assert len(r.best_position) == 2
    assert r.stop_reason == "max_iterations"
    assert r.evaluations > 0


def test_bounds_forms_equivalent():
    a = pso.minimize("sphere", bounds=(-5.0, 5.0), dim=3, seed=1)
    b = pso.minimize("sphere", bounds=[(-5.0, 5.0)] * 3, seed=1)
    assert a.best_position == b.best_position


def test_single_bound_without_dim_errors():
    import pytest

    with pytest.raises(ValueError):
        pso.minimize("sphere", bounds=(-5.0, 5.0))


def test_custom_objective_and_per_dimension_bounds():
    r = pso.minimize(
        lambda x: (x[0] - 1) ** 2 + (x[1] - 50) ** 2,
        bounds=[(-5, 5), (0, 100)],
        seed=1,
        max_iter=300,
    )
    assert abs(r.best_position[0] - 1) < 0.1
    assert abs(r.best_position[1] - 50) < 1.0


def test_integer_and_binary():
    r = pso.minimize(
        lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2,
        bounds=(-10, 10), dim=2, integer=True, seed=5,
    )
    assert r.best_position == [3.0, -2.0]


def test_mixed_var_types():
    r = pso.minimize(
        lambda x: (x[0] - 1.5) ** 2 + (x[1] - 3) ** 2 + (x[2] - 1) ** 2,
        bounds=[(-5, 5), (-10, 10), (0, 1)],
        var_types=["real", "integer", "binary"], seed=7,
    )
    assert r.best_position[1] == round(r.best_position[1])
    assert r.best_position[2] in (0.0, 1.0)


def test_constraints_penalty():
    # min x0^2 + x1^2 s.t. x0 + x1 >= 2  ->  optimum (1, 1).
    r = pso.minimize(
        lambda x: x[0] ** 2 + x[1] ** 2,
        bounds=(-5, 5), dim=2,
        constraints=[lambda x: 2 - x[0] - x[1]],
        seed=1, max_iter=300,
    )
    assert abs(sum(r.best_position) - 2) < 0.1


def test_callback_can_stop():
    seen = []

    def cb(it, best):
        seen.append(it)
        return it < 9

    r = pso.minimize("sphere", bounds=(-5, 5), dim=2, max_iter=1000, callback=cb, seed=1)
    assert r.stop_reason == "callback"
    assert len(seen) == 10


def test_vectorized_matches_scalar_shape():
    r = pso.minimize(
        lambda X: np.sum(np.asarray(X) ** 2, axis=1),
        bounds=(-5, 5), dim=4, vectorized=True, seed=0,
    )
    assert r.best_value < 1e-2


def test_topologies_and_variants():
    for velocity in ("inertia", "constriction", "fips"):
        for topology in ("global", "ring", "vonneumann", "random"):
            r = pso.minimize("sphere", bounds=(-5.12, 5.12), dim=2,
                             velocity=velocity, topology=topology, seed=1)
            assert r.best_value < 1.0


def test_multi_objective_pareto_front():
    front = pso.minimize_multi(
        lambda x: [sum(xi ** 2 for xi in x), sum((xi - 2) ** 2 for xi in x)],
        bounds=(-5, 5), dim=2, n_particles=60, max_iter=60, archive_size=40, seed=42,
    )
    assert len(front) >= 5
    objs = front.objectives
    # Mutually non-dominated.
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
    for a in objs:
        for b in objs:
            assert not dominates(a, b) or a == b


def test_benchmark_info():
    bound, optimum = pso.benchmark_info("ackley")
    assert bound == 32.768 and optimum == 0.0
