# Constraints

`turboswarm` handles inequality constraints with a **penalty method**: you pass
constraint functions, and solutions that violate them are penalized so the
swarm is pushed back into the feasible region.

Each constraint is a callable `g(x) -> float`, **feasible when `g(x) <= 0`**.
The optimizer adds a quadratic penalty to the objective:

$$f_\text{penalized}(x) = f(x) + \texttt{penalty} \cdot \sum_i \max(0,\, g_i(x))^2$$

## Example

Minimize `x₀² + x₁²` subject to `x₀ + x₁ ≥ 2` (rewritten as `2 - x₀ - x₁ ≤ 0`):

```python
import turboswarm as pso

r = pso.minimize(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5, 5)] * 2,
    constraints=[lambda x: 2 - x[0] - x[1]],
    seed=1,
)
print(r.best_position)   # ≈ [1.0, 1.0]  (the constrained optimum)
print(r.best_value)      # ≈ 2.0
```

Several constraints can be combined in the list. Tune the strength with
`penalty` (default `1e6`):

```python
r = pso.minimize(f, bounds=bounds,
                 constraints=[g1, g2, g3], penalty=1e4, seed=0)
```

!!! note
    Constraints require the objective to be a **Python callable** (not a native
    benchmark name), since the constraint functions are evaluated in Python.

## From Rust

The core stays constraint-agnostic: a constrained problem is just a penalized
objective, which you compose yourself.

```rust
let penalty = 1e6;
let objective = |x: &[f64]| {
    let g = 2.0 - x[0] - x[1];      // feasible when g <= 0
    let viol = g.max(0.0);
    x[0] * x[0] + x[1] * x[1] + penalty * viol * viol
};
```
