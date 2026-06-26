"""scikit-learn-compatible hyperparameter search powered by PSO.

``PSOSearchCV`` is a drop-in alternative to ``GridSearchCV`` /
``RandomizedSearchCV`` (or Optuna) that explores the hyperparameter space with
Particle Swarm Optimization::

    from sklearn.svm import SVC
    from turboswarm.integrations.sklearn import PSOSearchCV

    search = PSOSearchCV(
        SVC(),
        {"C": (1e-2, 1e2), "gamma": (1e-4, 1e0), "kernel": ["rbf", "poly"]},
        n_particles=20, max_iter=30, cv=5, random_state=0,
    )
    search.fit(X, y)
    search.best_params_, search.best_score_
    search.predict(X_new)

Search space per hyperparameter:
  - ``(low, high)`` float tuple  -> continuous range.
  - ``(low, high)`` int tuple    -> integer range.
  - ``[a, b, c]`` list           -> categorical choice.

Requires scikit-learn (``pip install turboswarm[sklearn]``). Importing this
module requires scikit-learn; ``import turboswarm`` does not.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils.validation import check_is_fitted

import turboswarm as _ts


def _encode_space(param_space):
    """Map a hyperparameter space to PSO bounds + var_types + per-dim decoders."""
    names, bounds, var_types, specs = [], [], [], []
    for name, spec in param_space.items():
        names.append(name)
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError(
                    f"range for '{name}' must be a (low, high) tuple, got {spec!r}"
                )
            low, high = spec
            ints = (isinstance(low, (int, np.integer))
                    and isinstance(high, (int, np.integer))
                    and not isinstance(low, bool))
            bounds.append((float(low), float(high)))
            var_types.append("integer" if ints else "real")
            specs.append({"kind": "int" if ints else "real"})
        elif isinstance(spec, list):
            if not spec:
                raise ValueError(f"categorical '{name}' has no choices")
            bounds.append((0.0, float(len(spec) - 1)))
            var_types.append("integer")
            specs.append({"kind": "cat", "choices": list(spec)})
        else:
            raise ValueError(
                f"'{name}': use a (low, high) tuple for a range or a list for "
                f"categorical choices, got {type(spec).__name__}"
            )
    return names, bounds, var_types, specs


def _decode(position, names, specs):
    params = {}
    for name, spec, value in zip(names, specs, position):
        if spec["kind"] == "real":
            params[name] = float(value)
        elif spec["kind"] == "int":
            params[name] = int(round(value))
        else:  # categorical: position is an index into choices
            idx = int(round(value))
            idx = max(0, min(idx, len(spec["choices"]) - 1))
            params[name] = spec["choices"][idx]
    return params


class PSOSearchCV(MetaEstimatorMixin, BaseEstimator):
    """Hyperparameter search over ``param_space`` using PSO, scoring by CV.

    Args:
        estimator: a scikit-learn estimator implementing ``fit``.
        param_space: dict mapping hyperparameter name to a ``(low, high)`` range
            (float -> continuous, int -> integer) or a ``list`` of categorical
            choices.
        n_particles: swarm size (candidate configurations per iteration).
        max_iter: number of PSO iterations.
        scoring: scikit-learn scorer (string or callable); higher is better.
        cv: cross-validation splits (int, splitter, or iterable).
        n_jobs: parallelism for ``cross_val_score``.
        refit: refit ``estimator`` on the whole dataset with the best params
            (enables ``predict``/``score``).
        random_state: PSO seed for reproducibility.
        velocity, topology: forwarded to :func:`turboswarm.minimize`.

    Attributes:
        best_params_, best_score_, best_estimator_, best_index_, cv_results_.
    """

    _required_parameters = ["estimator", "param_space"]

    def __init__(self, estimator, param_space, *, n_particles=20, max_iter=30,
                 scoring=None, cv=5, n_jobs=None, refit=True, random_state=None,
                 velocity="inertia", topology="global"):
        self.estimator = estimator
        self.param_space = param_space
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.refit = refit
        self.random_state = random_state
        self.velocity = velocity
        self.topology = topology

    def fit(self, X, y=None, **fit_params):
        names, bounds, var_types, specs = _encode_space(self.param_space)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        evaluated_params, mean_scores, std_scores = [], [], []

        def objective(position):
            params = _decode(position, names, specs)
            estimator = clone(self.estimator).set_params(**params)
            scores = cross_val_score(estimator, X, y, scoring=scorer, cv=cv,
                                     n_jobs=self.n_jobs)
            mean = float(np.mean(scores))
            evaluated_params.append(params)
            mean_scores.append(mean)
            std_scores.append(float(np.std(scores)))
            return -mean  # scorers follow "higher is better"; PSO minimizes

        result = _ts.minimize(
            objective, bounds=bounds, var_types=var_types,
            n_particles=self.n_particles, max_iter=self.max_iter,
            velocity=self.velocity, topology=self.topology,
            seed=self.random_state, record_history=False,
        )

        means = np.asarray(mean_scores)
        stds = np.asarray(std_scores)
        order = np.argsort(-means, kind="stable")
        ranks = np.empty(len(means), dtype=int)
        ranks[order] = np.arange(1, len(means) + 1)
        self.cv_results_ = {
            "params": evaluated_params,
            "mean_test_score": means,
            "std_test_score": stds,
            "rank_test_score": ranks,
        }
        self.best_index_ = int(np.argmax(means))
        self.best_params_ = _decode(result.best_position, names, specs)
        self.best_score_ = -result.best_value
        self.n_evaluations_ = result.evaluations

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(
                **self.best_params_).fit(X, y, **fit_params)
            if hasattr(self.best_estimator_, "classes_"):
                self.classes_ = self.best_estimator_.classes_
        return self

    def _check_refit(self):
        check_is_fitted(self, "best_params_")
        if not self.refit:
            raise AttributeError(
                "predict/score require refit=True (no best_estimator_ was fit)."
            )

    def predict(self, X):
        self._check_refit()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        self._check_refit()
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        self._check_refit()
        return self.best_estimator_.decision_function(X)

    def score(self, X, y=None):
        self._check_refit()
        scorer = check_scoring(self.best_estimator_, scoring=self.scoring)
        return scorer(self.best_estimator_, X, y)
