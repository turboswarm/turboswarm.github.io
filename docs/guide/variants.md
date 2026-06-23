# Velocity variants

A *variant* is a rule for updating each particle's velocity. In `pso-rs` a
variant is one implementation of the `Velocity` trait, selected by name from
Python (`velocity=`).

## Inertia (`"inertia"`)

Shi & Eberhart (1998). The classic rule:

$$v' = w\,v + c_1 r_1 (p_\text{best} - x) + c_2 r_2 (n_\text{best} - x)$$

- `w` — inertia weight (how much of the previous velocity is kept).
- `c1` — cognitive coefficient (pull toward the personal best).
- `c2` — social coefficient (pull toward the neighborhood best).

The inertia weight can decay linearly over the run (a classic
exploration/exploitation schedule), available from Rust via `with_decay`.

```python
r = pso.minimize("sphere", bounds=[(-5, 5)] * 2,
                 velocity="inertia", w=0.729, c1=1.49445, c2=1.49445, seed=1)
```

## Constriction (`"constriction"`)

Clerc & Kennedy (2002). Multiplies the whole update by a constriction factor χ
derived from the coefficients:

$$\chi = \frac{2}{\left|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}\right|}, \quad \varphi = c_1 + c_2 > 4$$

With the classic `c1 = c2 = 2.05` (φ = 4.1) you get χ ≈ 0.7298 — exactly the
default inertia weight `0.729`. Note that here χ is *derived* from a
convergence guarantee instead of being chosen by hand.

```python
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                 velocity="constriction", seed=1)
```

!!! note
    From Python, `"constriction"` derives χ from `c1 + c2`. If their sum does
    not exceed 4 (e.g. the defaults), it falls back to the classic 2.05/2.05.

## FIPS (`"fips"`)

Fully Informed Particle Swarm (Mendes, Kennedy & Neves, 2004). Instead of
listening to a single social source (the neighborhood best), the particle is
informed by **all** of its neighbors:

$$v' = \chi\left[ v + \sum_{k \in N} U\!\left(0, \tfrac{\varphi}{|N|}\right)(p_k - x) \right]$$

The total coefficient φ is split equally among the `|N|` neighbors, so the
expected social acceleration matches constriction — FIPS *redistributes* the
pull rather than increasing it. There is no separate cognitive term; the
particle's own personal best enters as one neighbor.

```python
# FIPS performs best with a LOCAL topology.
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                 velocity="fips", topology="ring", seed=1)
```

!!! tip
    FIPS with `"global"` makes every particle listen to everyone, so the swarm
    tends to collapse early. Pair it with `"ring"` or `"vonneumann"`.

See [Extending](../extending.md) to add your own variant.
