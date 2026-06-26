# Tuning a model's hyperparameters

This tutorial uses [`PSOSearchCV`](../guide/integrations.md#scikit-learn) to tune
a scikit-learn model the same way you would with `GridSearchCV` — but the swarm
searches the **continuous** hyperparameter space instead of a fixed grid.

```bash
pip install turboswarm[sklearn]
```

## The task

Classify the [wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
with an SVM. An SVM needs its features scaled, so we tune a `Pipeline` of
`StandardScaler` + `SVC`, optimizing three hyperparameters:

- `svc__C` — regularization, a wide **continuous** range.
- `svc__gamma` — kernel coefficient, a **continuous** range.
- `svc__kernel` — a **categorical** choice.

```python
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from turboswarm.integrations.sklearn import PSOSearchCV

X, y = load_wine(return_X_y=True)
pipe = Pipeline([("scale", StandardScaler()), ("svc", SVC())])

search = PSOSearchCV(
    pipe,
    {
        "svc__C": (1e-2, 1e3),          # continuous range  (float tuple)
        "svc__gamma": (1e-4, 1e0),      # continuous range
        "svc__kernel": ["rbf", "poly"], # categorical        (list)
    },
    n_particles=15, max_iter=15, cv=5, scoring="accuracy", random_state=0,
)
search.fit(X, y)

print(search.best_params_)
# {'svc__C': 907.1..., 'svc__gamma': 0.0664..., 'svc__kernel': 'rbf'}
print(f"{search.best_score_:.4f}")     # 0.9889  (5-fold CV accuracy)
print(search.n_evaluations_)           # 240 configurations evaluated
```

`PSOSearchCV` is a regular scikit-learn estimator: after `fit` it refits the best
configuration on the whole dataset, so you can predict directly.

```python
search.predict(X[:5])
search.score(X, y)
```

## How it did vs. a grid

A comparable `GridSearchCV` over a discrete grid reaches the **same** CV
accuracy here:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    pipe,
    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1],
     "svc__kernel": ["rbf"]},
    cv=5,
).fit(X, y)

print(f"{grid.best_score_:.4f}")        # 0.9889
```

The difference is *how* they search: the grid only ever tries the values you
list, while PSO moves through the continuous `C`/`gamma` space and can land on
values no grid contains (here `C ≈ 907`, `gamma ≈ 0.066`). On larger or
higher-dimensional spaces — where a full grid explodes combinatorially — that is
where PSO pays off.

## Inspecting the search

Every evaluated configuration is recorded in `cv_results_`, ready for pandas:

```python
import pandas as pd

df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
df.head()        # columns: params, mean_test_score, std_test_score, rank_test_score
```

## Next steps

- Swap in your own estimator and space — integer ranges use `(low, high)` with
  **int** endpoints (e.g. `"max_depth": (1, 20)`).
- For very expensive objectives, distribute evaluations with the
  [Joblib/Dask integration](../guide/integrations.md#joblib-dask).
