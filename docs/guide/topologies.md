# Topologies

A *topology* is the social structure of the swarm: who informs whom. In
`pso-rs` it is one implementation of the `Topology` trait, selected by name
(`topology=`). A topology is fundamentally defined by its **neighborhood**:
`neighbors(i)` returns the indices that inform particle `i` (including itself);
`best_neighbor` is derived from it by default.

The neighborhood controls how fast good information spreads. Faster spreading
converges quicker but risks premature convergence on local optima.

## Global (`"global"`)

The gbest topology: every particle sees the best of the whole swarm. Fastest
information flow, highest risk of getting stuck on a local optimum. The default.

```python
r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, topology="global")
```

## Ring (`"ring"`)

The lbest topology: each particle only sees its immediate neighbors on a
circle (one on each side by default). Information travels slowly around the
ring, which preserves diversity and helps on multimodal problems, at the cost
of slower convergence.

```python
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, topology="ring")
```

## Von Neumann (`"vonneumann"`)

Particles sit on a 2D `rows × cols` grid; each is informed by its four
neighbors (up, down, left, right) with toroidal (wrap-around) edges. It is a
middle ground between the ring (very local) and global, and often performs well
on multimodal problems.

From Python the grid is sized automatically to be as square as possible for the
swarm size, so `rows · cols ≈ n_particles` (use a perfect square like 49 to
fill the grid exactly).

```python
r = pso.minimize("ackley", bounds=[(-32.768, 32.768)] * 2,
                 topology="vonneumann", n_particles=49)
```

## Random (`"random"`)

Each particle is informed by a handful of randomly chosen neighbors that change
every iteration. There is no fixed structure, which keeps information flow
diverse. The topology owns a seeded RNG, so runs stay reproducible (it does not
disturb the optimizer's own RNG stream).

```python
r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                 topology="random", seed=1)
```

See [Extending](../extending.md) to add your own topology.
