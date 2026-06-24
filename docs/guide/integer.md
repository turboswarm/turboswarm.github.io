# Integer optimization

`pso-rs` optimizes integer variables with a deliberately simple approach:
**the swarm always moves in a continuous internal space, and
discretization happens only at evaluation time** (in `SearchSpace::decode`).

This keeps the central question of integer PSO — *how do you discretize?* —
explicit, and lets you compare strategies easily. It is
also why the core never needs generics over the position type: positions are
always `Vec<f64>` internally.

## From Python

Pass `integer=True`. Bounds are still given as numbers; they are rounded to
integer limits.

```python
import turboswarm as pso

# f(x) = (x0 - 3)^2 + (x1 + 2)^2, integer optimum at (3, -2).
r = pso.minimize(
    lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2,
    bounds=[(-10, 10)] * 2,
    integer=True,
    seed=5,
)
print(r.best_position)   # [3.0, -2.0]
print(r.best_value)      # 0.0
```

The objective receives already-decoded integer values. The returned
`best_position` corresponds to integer coordinates.

## Binary variables

For `{0, 1}` problems (feature selection, knapsack, subset choice) pass
`binary=True`. It is the binary special case of the integer space; the
dimension is taken from `bounds`.

```python
# 0/1 knapsack: pick items maximizing value within a weight cap.
values, weights, cap = [10, 13, 18, 31, 7, 15], [2, 3, 4, 7, 1, 3], 10
r = pso.minimize(
    lambda x: -sum(v * xi for v, xi in zip(values, x)),     # minimize -value
    bounds=[(0, 1)] * len(values),
    binary=True,
    constraints=[lambda x: sum(w * xi for w, xi in zip(weights, x)) - cap],
    n_particles=50, max_iter=200, seed=1,
)
chosen = [int(b) for b in r.best_position]
```

## Mixed variable types

Real problems often mix continuous, integer and binary variables. Pass
`var_types` — one of `"real"`, `"integer"` or `"binary"` per dimension (same
length as `bounds`):

```python
# x0 continuous, x1 integer, x2 binary.
r = pso.minimize(
    lambda x: (x[0] - 1.5) ** 2 + (x[1] - 3) ** 2 + (x[2] - 1) ** 2,
    bounds=[(-5, 5), (-10, 10), (0, 1)],
    var_types=["real", "integer", "binary"],
    seed=7,
)
# r.best_position == [1.5, 3.0, 1.0]  (integer/binary dims are whole-valued)
```

Integer and binary dimensions come back as whole-valued floats. `var_types`
takes precedence over the `integer` / `binary` flags. In Rust this is the
`MixedSpace` (its `Scalar` is `f64`).

## Discretization strategies (Rust)

From Rust, `IntegerSpace` supports three strategies via `with_discretization`:

- `Round` — nearest integer (default, most common).
- `Truncate` — toward zero.
- `Floor` — round down.

```rust
use turboswarm_core::prelude::*;

let space = IntegerSpace::new(vec![(-10, 10); 2])
    .with_discretization(Discretization::Floor);
```

Comparing how the chosen strategy affects convergence on the same problem is a
compact way to compare them.
