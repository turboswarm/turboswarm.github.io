# Integer and mixed optimization

PSO moves through a continuous space internally, but `turboswarm` can optimize
**integer**, **binary** and **mixed** variables by discretizing only at
evaluation time. This tutorial solves a classic 0/1 knapsack, then a problem
that mixes variable types.

## A 0/1 knapsack

Pick a subset of items that maximizes total value without exceeding a weight
cap — a binary problem with one constraint.

```python
import turboswarm as pso

values  = [10, 13, 18, 31, 7, 15]
weights = [2,  3,  4,  7, 1,  3]
cap = 10

result = pso.minimize(
    # maximize value  ->  minimize negative value
    lambda x: -sum(v * xi for v, xi in zip(values, x)),
    bounds=[(0, 1)] * len(values),
    binary=True,                                       # variables are {0, 1}
    constraints=[                                      # g(x) <= 0
        lambda x: sum(w * xi for w, xi in zip(weights, x)) - cap
    ],
    n_particles=50, max_iter=200, seed=1,
)

chosen = [int(b) for b in result.best_position]
print(chosen)                                          # [1, 0, 1, 0, 1, 1]
print(sum(v for v, c in zip(values, chosen) if c))     # value  = 50
print(sum(w for w, c in zip(weights, chosen) if c))    # weight = 10  (== cap)
```

`binary=True` makes each variable a `{0, 1}` choice; the constraint is written in
the `g(x) ≤ 0` form (here *total weight − cap*). The returned `best_position`
comes back as whole-valued floats, so we cast to `int`.

### Did it find the true optimum?

This instance is small enough to brute-force, which confirms PSO found the
**global** optimum:

```python
import itertools

best = max(
    (combo for combo in itertools.product([0, 1], repeat=len(values))
     if sum(w * c for w, c in zip(weights, combo)) <= cap),
    key=lambda combo: sum(v * c for v, c in zip(values, combo)),
)
print(sum(v * c for v, c in zip(values, best)))        # 50  (same value)
```

## Mixed variable types

Real problems often mix continuous, integer and binary variables. Pass
`var_types` — one of `"real"`, `"integer"` or `"binary"` per dimension:

```python
result = pso.minimize(
    lambda x: (x[0] - 1.5) ** 2 + (x[1] - 3) ** 2 + (x[2] - 1) ** 2,
    bounds=[(-5, 5), (-10, 10), (0, 1)],
    var_types=["real", "integer", "binary"],
    seed=7,
)
print(result.best_position)        # [1.5, 3.0, 1.0]
print(f"{result.best_value:.4f}")  # 0.0000
```

The continuous dimension settles on `1.5`, while the integer and binary
dimensions come back whole-valued (`3.0`, `1.0`). The key idea: the swarm always
searches a continuous space, and the integer/binary nature is applied only when
the objective is evaluated — so every variant, topology and constraint works
unchanged across variable types.

## Next steps

- For the discretization strategies (round/truncate/floor) and the design
  rationale, see the [Integer optimization guide](../guide/integer.md).
- Combine integer/mixed variables with multiple objectives using
  [`minimize_multi`](multi-objective.md).
