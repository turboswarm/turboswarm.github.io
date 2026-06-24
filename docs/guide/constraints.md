# Constraints

`turboswarm` handles constraints with a **penalty method**: you pass constraint
functions, and solutions that violate them are penalized so the swarm is pushed
back into the feasible region. For hard constraints you can also supply a
**repair operator** that maps each candidate back into the feasible set.

## Inequality constraints

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

## Equality constraints

Equality constraints are callables `h(x) -> float`, **feasible when `h(x) == 0`**,
passed as `equality_constraints=`. They contribute a squared-deviation penalty:

$$f_\text{penalized}(x) = f(x) + \texttt{penalty} \cdot \sum_j h_j(x)^2$$

Minimize `x₀² + x₁²` subject to `x₀ + x₁ = 2` (optimum `(1, 1)`):

```python
r = pso.minimize(
    lambda x: x[0]**2 + x[1]**2,
    bounds=[(-5, 5)] * 2,
    equality_constraints=[lambda x: x[0] + x[1] - 2.0],
    penalty=1e4, seed=1,
)
print(r.best_position)   # ≈ [1.0, 1.0]
```

!!! warning "Tuning `penalty` for equalities"
    Equality penalties are sensitive to `penalty`. A very large value (such as
    the `1e6` default) makes the feasible valley so steep that the swarm
    collapses onto the *first* feasible point it finds — feasible but
    sub-optimal. Moderate weights (`1e3`–`1e4`) usually balance feasibility and
    objective much better. For hard equalities, a `repair` operator is often
    more robust.

## Repair operator

A `repair` is a callable `repair(x) -> x'` applied to every candidate **before**
it is evaluated, mapping infeasible points back into (or towards) the feasible
region. The objective and the constraints see the repaired point, and the
returned `best_position` is repaired too, so the reported solution is consistent
with its value. This enforces a constraint *exactly* instead of just penalizing
it.

Optimize on the simplex `x₀ + x₁ = 1` by projecting onto it:

```python
def repair(x):
    s = sum(x) or 1.0
    return [xi / s for xi in x]            # project onto sum(x) == 1

r = pso.minimize(
    lambda x: (x[0] - 0.7)**2 + (x[1] - 0.3)**2,
    bounds=[(0, 1)] * 2, repair=repair, seed=1,
)
print(r.best_position, sum(r.best_position))   # ≈ [0.7, 0.3]  1.0
```

`repair` can be combined with `constraints` and `equality_constraints` (repair
runs first, then any remaining violation is penalized).

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
