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


# --- CEC-family functions (canonical, unshifted; global minimum 0) ---


def bent_cigar(x):
    """Bent Cigar: unimodal, severely ill-conditioned. Minimum f(0) = 0."""
    if not x:
        return 0.0
    return x[0] ** 2 + 1e6 * sum(xi * xi for xi in x[1:])


def discus(x):
    """Discus: one expensive direction (1e6), the rest cheap. Minimum f(0) = 0."""
    if not x:
        return 0.0
    return 1e6 * x[0] ** 2 + sum(xi * xi for xi in x[1:])


def elliptic(x):
    """High Conditioned Elliptic: condition number 1..1e6. Minimum f(0) = 0."""
    n = len(x)
    if n <= 1:
        return x[0] ** 2 if x else 0.0
    return sum(1e6 ** (i / (n - 1)) * xi * xi for i, xi in enumerate(x))


def zakharov(x):
    """Zakharov: unimodal, no local minima. Minimum f(0) = 0."""
    sum_sq = sum(xi * xi for xi in x)
    half = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum_sq + half ** 2 + half ** 4


def levy(x):
    """Levy: multimodal. Global minimum f(1, ..., 1) = 0 (away from origin)."""
    if not x:
        return 0.0
    w = [1 + (xi - 1) / 4 for xi in x]
    n = len(w)
    term1 = math.sin(math.pi * w[0]) ** 2
    term_mid = sum(
        (w[i] - 1) ** 2 * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2)
        for i in range(n - 1)
    )
    term_last = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
    return term1 + term_mid + term_last


def expanded_schaffer(x):
    """Expanded Schaffer F6: deceptive, highly multimodal. Minimum f(0) = 0."""
    def g(a, b):
        s = a * a + b * b
        return 0.5 + (math.sin(math.sqrt(s)) ** 2 - 0.5) / (1 + 0.001 * s) ** 2

    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return g(x[0], x[0])
    return sum(g(x[i], x[(i + 1) % n]) for i in range(n))


#: Recommended symmetric bound per dimension for each benchmark.
BOUNDS = {
    "sphere": 5.12,
    "rastrigin": 5.12,
    "rosenbrock": 2.048,
    "ackley": 32.768,
    "griewank": 600.0,
    "schwefel": 500.0,
    "bent_cigar": 100.0,
    "discus": 100.0,
    "elliptic": 100.0,
    "zakharov": 10.0,
    "levy": 10.0,
    "expanded_schaffer": 100.0,
}
