"""Test functions in pure Python (mirror of the Rust ones).

Useful for the notebooks before having the native core compiled.
"""
import math


def sphere(x):
    """Sphere: convex, unimodal. Global minimum f(0) = 0."""
    return sum(xi * xi for xi in x)


def rastrigin(x):
    """Rastrigin: highly multimodal. Global minimum f(0) = 0."""
    return 10 * len(x) + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(x):
    """Rosenbrock: narrow banana valley. Global minimum f(1, ..., 1) = 0."""
    return sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def ackley(x):
    """Ackley: nearly flat far out with a narrow well. Minimum f(0) = 0."""
    n = len(x)
    sum_sq = sum(xi * xi for xi in x)
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20
        + math.e
    )


def griewank(x):
    """Griewank: many regular local minima. Global minimum f(0) = 0."""
    s = sum(xi * xi for xi in x) / 4000
    p = 1.0
    for i, xi in enumerate(x):
        p *= math.cos(xi / math.sqrt(i + 1))
    return 1 + s - p


def schwefel(x):
    """Schwefel: multimodal, optimum FAR from the origin (≈420.97/dim).

    Global minimum f(420.9687, ...) = 0.
    """
    return 418.9828872724338 * len(x) - sum(
        xi * math.sin(math.sqrt(abs(xi))) for xi in x
    )


#: Recommended symmetric bound per dimension for each benchmark.
BOUNDS = {
    "sphere": 5.12,
    "rastrigin": 5.12,
    "rosenbrock": 2.048,
    "ackley": 32.768,
    "griewank": 600.0,
    "schwefel": 500.0,
}
