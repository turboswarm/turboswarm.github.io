# Constrained optimization

Real problems usually come with constraints. This tutorial solves a constrained
minimization with `turboswarm`'s penalty method and verifies the result against
the known optimum.

## The problem

Minimize a quadratic subject to one inequality constraint:

$$\min_{x}\; x_0^2 + x_1^2 \quad\text{subject to}\quad x_0 + x_1 \ge 1$$

The unconstrained minimum is the origin, but it is infeasible. The constrained
optimum sits on the line $x_0 + x_1 = 1$ closest to the origin: $(0.5,\ 0.5)$,
where $f = 0.5$.

## Express the constraints

`turboswarm` expects each constraint in the form $g(x) \le 0$. Rewrite
$x_0 + x_1 \ge 1$ as:

$$g(x) = 1 - x_0 - x_1 \le 0$$

Pass the constraints as a list of callables; infeasible particles are penalized
(the default `penalty` weight is large):

```python
import turboswarm as pso

result = pso.minimize(
    lambda x: x[0] ** 2 + x[1] ** 2,            # objective
    bounds=[(-5, 5)] * 2,
    constraints=[lambda x: 1.0 - x[0] - x[1]],  # g(x) = 1 - x0 - x1 <= 0
    seed=0, n_particles=40, max_iter=200,
)

print(result.best_position)          # [0.5, 0.5]
print(f"{result.best_value:.4f}")    # 0.5000
print(result.best_position[0] + result.best_position[1])   # 1.0  (constraint active)
```

The swarm lands on $(0.5, 0.5)$ with $f = 0.5$, exactly the analytical optimum,
and the constraint is active ($x_0 + x_1 = 1$).

## Multiple constraints

Add as many as you need — each is one callable returning $g(x) \le 0$. For
example, also requiring $x_0 \le 0.4$:

```python
result = pso.minimize(
    lambda x: x[0] ** 2 + x[1] ** 2,
    bounds=[(-5, 5)] * 2,
    constraints=[
        lambda x: 1.0 - x[0] - x[1],   # x0 + x1 >= 1
        lambda x: x[0] - 0.4,          # x0 <= 0.4
    ],
    seed=0, n_particles=40, max_iter=300,
)
print(result.best_position)          # ~[0.4, 0.6]
```

## Tuning the penalty

The penalty weight trades off objective quality against constraint satisfaction.
If your solution drifts outside the feasible region, raise `penalty`; if the
optimizer is too conservative, lower it:

```python
result = pso.minimize(
    objective, bounds=bounds, constraints=constraints,
    penalty=1e8,        # stricter feasibility (default 1e6)
    seed=0,
)
```

For **equality** constraints, an exact `repair` operator, and the theory behind
the penalty method, see the [Constraints guide](../guide/constraints.md).

## Next steps

- Mix continuous and integer variables in a constrained problem with
  [`var_types`](../guide/integer.md#mixed-variable-types).
- Wrap an expensive constrained objective for parallel evaluation via the
  [Joblib/Dask integration](../guide/integrations.md#joblib-dask).
