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
Python API. PSO is a population-based, gradient-free optimization method in
which a swarm of candidate solutions ("particles") moves through the search
space guided by each particle's own best position and the best position found
in its neighbourhood. `turboswarm` targets real, integer, mixed, binary and
interval-valued ("grey") decision variables, and exposes established
algorithmic variants, swarm topologies, constraint handling, multi-objective
optimization and visualization behind a single, uniform interface. The same
Rust core is usable directly from Rust (through zero-cost generics) and from
Python (through PyO3 and maturin), where objective functions may be supplied
either as Python callables or selected from native benchmarks that run without
holding the Global Interpreter Lock (GIL). Every experiment is seedable, so
results are reproducible.

# Statement of need

PSO is one of the most widely used metaheuristics for continuous and
combinatorial optimization, with applications across engineering, machine
learning and the sciences. Researchers and students need an implementation that
is fast, reproducible, broad in scope, and easy to extend with new variants for
comparative studies. Existing Python libraries cover parts of this need, but
they are implemented predominantly in pure Python, which ties the cost of the
optimization loop to the interpreter and makes large or repeated experiments
slow. `turboswarm` was written to provide a compiled, reproducible and
extensible PSO toolkit that is nonetheless as easy to call as a pure-Python
package, suitable for research experiments, algorithm benchmarking and teaching.

# State of the field

Several mature open-source PSO and metaheuristics packages exist, including
`pyswarms` [@miranda2018pyswarms], `pymoo` [@blank2020pymoo], `DEAP`
[@fortin2012deap] and `nevergrad` [@rapin2018nevergrad]. These are valuable and
widely adopted, but they are implemented mainly in pure Python, so the swarm
loop itself runs in the interpreter. `turboswarm` differs in three ways that
motivated building a new library rather than contributing to an existing one.
First, its optimization loop and native objectives are compiled (Rust) and run
without the GIL, giving a measured speed advantage (see Performance). Second, it
treats extensibility as a first-class design goal: a new variant is a
self-contained trait implementation, not a fork of the core (see Software
design). Third, it provides a grey-number (interval-valued) search space, in
which each decision variable is known only to lie within an interval — a
capability we are not aware of in the libraries above. Together these make
`turboswarm` complementary to, rather than a re-implementation of, the existing
ecosystem.

# Software design

The optimization loop is fully decoupled from any concrete variant through three
traits. `SearchSpace` defines the domain; the only difference between real and
integer variables lives in its `decode` step, so the swarm always operates on a
continuous internal representation and the discretization happens only at
evaluation time. `Velocity` defines the update rule, so a single PSO variant
(inertia [@shi1998modified], constriction [@clerc2002particle] or the Fully
Informed Particle Swarm [@mendes2004fully]) is exactly one implementation of
this trait. `Topology` defines the social structure of the swarm (global, ring,
Von Neumann or random). Because the loop is generic over these traits, adding a
variant requires no change to the core and is accompanied by a convergence test
against a known optimum. For the Python boundary the same traits are implemented
for boxed trait objects, so variants can be selected at runtime by name without
duplicating the loop. This design confines the real/integer/mixed/binary/grey
distinction to the search space and makes the library a convenient base for
comparing variants under identical conditions.

# Features

- **Velocity variants:** inertia [@shi1998modified], constriction
  [@clerc2002particle] and Fully Informed PSO [@mendes2004fully].
- **Topologies:** global (gbest), ring (lbest), Von Neumann and random.
- **Search spaces:** continuous, integer, binary, mixed (per-dimension type)
  and grey (interval-valued variables).
- **Multi-objective (MOPSO):** external Pareto archive with crowding distance
  or an adaptive grid for diversity, a turbulence operator, and a hypervolume
  quality indicator [@coello2004handling].
- **Constraints:** inequality and equality constraints via a penalty, plus an
  optional repair operator.
- **Run control:** stop on target value, evaluation budget, wall-clock budget
  or stagnation, with a per-iteration callback and a reported stop reason.
- **Performance:** velocity clamping, rayon-based parallel evaluation and a
  NumPy-vectorized batch path.
- **Visualization (Python):** convergence, variant comparison, 2D swarm
  animation, Pareto-front and hyperparameter-sensitivity plots.

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

# Research impact

`turboswarm` is distributed on PyPI (`turboswarm`) and crates.io
(`turboswarm-core`) and archived on Zenodo (DOI 10.5281/zenodo.20832446), so it
can be installed and cited directly. By exposing many PSO variants, topologies
and search spaces behind one reproducible interface, it lowers the cost of
comparative algorithmic studies and supports teaching through its built-in
visualization and animation. Its trait-based core is also the foundation for
ongoing methodological work on optimization with grey (interval-valued)
variables. The project ships automated tests, continuous integration for both
the Rust core and the Python layer, narrative and API documentation, and
contribution guidelines, so it is ready for external use and community
contributions.

# AI usage disclosure

The `turboswarm` library itself — the Rust core, the Python bindings and the
algorithms — was designed and written by the author. Generative AI (Anthropic's
Claude, used through Claude Code) assisted with drafting this paper,
implementing the comparative benchmark suite, and editing parts of the
documentation. All AI-assisted text and code were reviewed, tested and validated
by the author, who takes full responsibility for the content.

# Acknowledgements

The author received no specific funding for this work, and acknowledges the
open-source Rust and PyO3 communities, whose tooling underpins the Rust–Python
boundary used here.

# References
