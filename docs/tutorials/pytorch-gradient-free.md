# Gradient-free training of a neural network

Backpropagation needs a differentiable loss. When the thing you actually care
about is **not** differentiable — classification accuracy, a hard threshold, a
simulator's output, a discrete reward — PSO optimizes it directly. This tutorial
trains a small PyTorch network by optimizing its weights with `turboswarm`,
using **accuracy** (a step function, no gradient) as the objective.

```bash
pip install torch        # example only; turboswarm has no torch dependency
```

## The setup

A toy XOR-like problem (the two classes are separated by `x0 * x1 > 0`), which a
linear model cannot solve — it needs the hidden layer's nonlinearity.

```python
import numpy as np
import torch
import torch.nn as nn
import turboswarm as pso

rng = np.random.RandomState(0)
X = rng.uniform(-2, 2, size=(200, 2)).astype(np.float32)
y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)        # XOR-like
Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
```

## Flatten the weights into a search vector

PSO searches a flat `Vec<f64>`, so we map between the swarm's position vector and
the model's parameter tensors. The model here has 42 parameters.

```python
shapes = [p.shape for p in model.parameters()]
sizes  = [p.numel() for p in model.parameters()]
n_params = sum(sizes)        # 42

def set_params(vec):
    i = 0
    with torch.no_grad():
        for p, shape, size in zip(model.parameters(), shapes, sizes):
            p.copy_(torch.tensor(vec[i:i + size], dtype=torch.float32).view(shape))
            i += size
```

## Optimize a non-differentiable objective

The objective sets the weights, runs a forward pass, and returns **negative
accuracy** (PSO minimizes). There is no gradient anywhere — `argmax` and the
equality count are not differentiable, which is exactly the point.

```python
def neg_accuracy(vec):
    set_params(vec)
    with torch.no_grad():
        pred = model(Xt).argmax(dim=1)
        return -(pred == yt).float().mean().item()

result = pso.minimize(
    neg_accuracy,
    bounds=[(-3, 3)] * n_params,
    n_particles=60, max_iter=200, seed=0,
)

print(f"{-result.best_value:.3f}")     # 1.000  — perfect accuracy
set_params(result.best_position)        # load the best weights back into the model
```

The swarm finds weights that classify the XOR-like data perfectly (accuracy
`1.000`), without ever computing a gradient.

## When this is useful

This is not a replacement for SGD on large differentiable models — gradients are
far more efficient there. PSO earns its place when the objective is **not
differentiable or not available in closed form**:

- optimizing a **discrete metric** directly (accuracy, F1, a business KPI),
- tuning **non-differentiable components** (thresholds, routing, architecture
  choices) of a pipeline,
- black-box settings where you only observe the output (simulators, RL returns).

The same pattern works with any framework (TensorFlow, JAX, NumPy) — only the
forward pass inside the objective changes. For tuning *hyperparameters* rather
than weights, use [`PSOSearchCV`](hyperparameter-tuning.md) instead.
