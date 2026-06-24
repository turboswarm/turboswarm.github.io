---
title: 'turboswarm: A general-purpose, extensible Particle Swarm Optimization library with a Rust core'
tags:
  - Rust
  - Python
  - optimization
  - particle swarm optimization
  - metaheuristics
  - multi-objective optimization
authors:
  - name: Jose L. Salmeron
    orcid: 0000-0001-7811-3716
    affiliation: 1
affiliations:
  - name: CUNEF Universidad, Madrid, Spain
    index: 1
date: 24 June 2026
bibliography: paper.bib
---

# Summary

`turboswarm` is a general-purpose Particle Swarm Optimization (PSO) library
[@kennedy1995particle] with a compute core written in Rust and an ergonomic
Python API. It targets real, integer, mixed and binary decision variables, and
is built as an *extensible framework*: each algorithmic variant is a single
trait implementation, so adding a new velocity rule or swarm topology does not
require touching the optimization loop. The same Rust core is usable directly
from Rust (zero-cost generics) and from Python (via PyO3 + maturin), where
objectives may be supplied either as Python callables or selected from native
benchmarks that run without holding the Global Interpreter Lock. `turboswarm`
ships several established variants (inertia [@shi1998modified], constriction
[@clerc2002particle] and the Fully Informed Particle Swarm
[@mendes2004fully]), swarm topologies (global, ring, Von Neumann and random),
inequality and equality constraints with an optional repair operator,
multi-objective optimization through an external Pareto archive
[@coello2004handling], and a `matplotlib`-based visualization module for
convergence curves, variant comparison, 2D swarm animation and Pareto fronts.
Every experiment is seedable for reproducibility.

# Statement of need

PSO is one of the most widely used metaheuristics for continuous and
combinatorial optimization, and several mature Python libraries already exist,
including `pyswarms` [@miranda2018pyswarms], `pymoo` [@blank2020pymoo], `DEAP`
[@fortin2012deap] and `nevergrad` [@rapin2018nevergrad]. These are, however,
implemented predominantly in pure Python, which couples the cost of the
optimization loop itself to the interpreter. `turboswarm` addresses two gaps.

First, **performance without sacrificing usability**: the swarm loop, the
search-space machinery and the native benchmarks run in compiled Rust, while
the public API stays in Python. For objectives that can be expressed natively
the entire optimization runs without the GIL, and rayon-backed parallel and
NumPy-vectorized evaluation paths are provided for expensive Python
objectives.

Second, **extensibility as a first-class concern**: the optimization loop is
fully decoupled from the concrete variant through three traits — `SearchSpace`
(the domain, where the only difference between real and integer variables
lives, at decode time), `Velocity` (the update rule) and `Topology` (the
social structure). A new PSO variant is therefore a self-contained trait
implementation accompanied by a convergence test, rather than a fork of the
core loop. This makes the library a convenient base both for teaching and for
comparing algorithmic variants under identical conditions.

Beyond the standard real/integer/mixed/binary spaces, `turboswarm` also
provides a **grey-number search space**, in which each decision variable is an
interval-valued (grey) quantity rather than a crisp scalar — a feature not
available in the libraries above. This supports optimization under variables
known only within bounds, and is exercised by a dedicated convergence test and
benchmark.

The combination of a compiled core, an extensible trait-based design, broad
feature coverage (constraints, multi-objective optimization, several spaces and
topologies) and built-in visualization makes `turboswarm` suitable for
research, benchmarking and teaching contexts where both reproducibility and
inspection of the swarm dynamics matter.

# Performance

To illustrate the benefit of the compiled core we compare `turboswarm` against
`pyswarms` [@miranda2018pyswarms], `pyswarm` and `pymoo` [@blank2020pymoo] on
five standard functions (sphere, rastrigin, ackley, rosenbrock, griewank) and
three dimensions (2, 10, 30), using an identical swarm configuration for every
library (40 particles, 200 iterations, $w = 0.729$, $c_1 = c_2 = 1.49445$;
median over five seeds after a warm-up). Through its native objective path
`turboswarm` is the fastest library across all functions and dimensions tested.
\autoref{fig:speedup} shows its speedup over `pyswarms`, the fastest
competitor: roughly $3.5\times$ at two dimensions, narrowing towards $2\times$
at thirty dimensions as the vectorized NumPy objective amortizes more of the
per-iteration overhead. Solution quality was comparable across libraries;
`turboswarm` reached the best objective in several cases, while `pymoo` did so
on a few high-dimensional multimodal problems.

![Speedup of `turboswarm` (native objective path) over `pyswarms` across
dimensions, for five benchmark functions.\label{fig:speedup}](benches/results/speedup.png)

Measured on an Apple M5 (10 cores, macOS 26.5). Absolute numbers are
machine-dependent; the full benchmark — including the raw results and machine
provenance — is reproducible with `benches/bench_suite.py`.

# Features

- **Velocity variants:** inertia [@shi1998modified], constriction
  [@clerc2002particle] and Fully Informed PSO [@mendes2004fully].
- **Topologies:** global (gbest), ring (lbest), Von Neumann and random.
- **Search spaces:** continuous, integer, binary, mixed (per-dimension type)
  and grey (interval-valued variables).
- **Multi-objective (MOPSO):** external Pareto archive with crowding distance
  or an adaptive grid for diversity, a turbulence operator, and a hypervolume
  quality indicator.
- **Constraints:** inequality and equality constraints via a penalty, plus an
  optional repair operator.
- **Run control:** stop on target value, evaluation budget, wall-clock budget
  or stagnation, with a per-iteration callback and a reported stop reason.
- **Performance:** velocity clamping, rayon-based parallel evaluation and a
  NumPy-vectorized batch path.
- **Visualization (Python):** convergence, variant comparison, 2D swarm
  animation, Pareto-front and hyperparameter-sensitivity plots.

# Acknowledgements

We acknowledge the open-source Rust and PyO3 communities, whose tooling
underpins the Rust–Python boundary used in this work.

# References
