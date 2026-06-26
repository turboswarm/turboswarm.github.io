# Tutorials

Hands-on, end-to-end walkthroughs of real tasks. Each one is self-contained and
runnable top to bottom. If you are new, start with
[Getting started](../getting-started.md) first.

- **[Tuning a model's hyperparameters](hyperparameter-tuning.md)** — use
  `PSOSearchCV` as a drop-in alternative to `GridSearchCV`, tune an SVM on a
  real dataset, and read the results.
- **[Visualizing and animating a swarm](visualizing-convergence.md)** — record a
  run, plot convergence, compare variants, and animate the swarm over the
  objective landscape.
- **[Constrained optimization](constrained-optimization.md)** — solve a problem
  with inequality constraints using the penalty method and verify the optimum.
- **[Multi-objective optimization](multi-objective.md)** — find and visualize a
  Pareto front of conflicting objectives with MOPSO, and measure it with
  hypervolume.
- **[Integer and mixed optimization](integer-mixed.md)** — solve a 0/1 knapsack
  (and verify the global optimum) and optimize real/integer/binary variables
  together.

Each tutorial only needs `turboswarm` plus the extras it calls out (e.g.
`pip install turboswarm[sklearn]` for the first one).
