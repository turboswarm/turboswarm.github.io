"""Use Particle Swarm Optimization as an Optuna sampler.

``TurboswarmSampler`` drives an Optuna study with a PSO swarm: each Optuna trial
evaluates one particle, and after a full generation the swarm updates its
velocities from the trials' objective values. Drop it into a study like any
other sampler::

    import optuna
    from turboswarm.integrations.optuna import TurboswarmSampler

    study = optuna.create_study(sampler=TurboswarmSampler(n_particles=20, seed=0))
    study.optimize(objective, n_trials=200)
    study.best_params, study.best_value

It supports ``Float``/``Int`` (including ``log``) and ``Categorical``
distributions, and both ``minimize`` and ``maximize`` study directions. It is
designed for sequential studies (``n_jobs=1``).

Requires Optuna (``pip install turboswarm[optuna]``). Importing this module
requires Optuna; ``import turboswarm`` does not.
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import TrialState


def _decode(u, dist):
    """Map a coordinate u in [0, 1] to a value of the given distribution."""
    u = float(min(max(u, 0.0), 1.0))
    if isinstance(dist, CategoricalDistribution):
        idx = min(int(u * len(dist.choices)), len(dist.choices) - 1)
        return dist.choices[idx]
    if isinstance(dist, (FloatDistribution, IntDistribution)):
        low, high = dist.low, dist.high
        if getattr(dist, "log", False):
            value = float(low) * (float(high) / float(low)) ** u
        else:
            value = low + u * (high - low)
        if isinstance(dist, IntDistribution):
            step = dist.step or 1
            value = low + round((value - low) / step) * step
            return int(min(max(value, low), high))
        if dist.step:
            value = low + round((value - low) / dist.step) * dist.step
        return float(min(max(value, low), high))
    raise ValueError(f"unsupported distribution: {type(dist).__name__}")


class TurboswarmSampler(BaseSampler):
    """An Optuna sampler that searches with a PSO swarm.

    Args:
        n_particles: swarm size (Optuna trials per generation).
        w, c1, c2: PSO inertia / cognitive / social coefficients.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, n_particles=20, *, w=0.729, c1=1.49445, c2=1.49445,
                 seed=None):
        self.n_particles = n_particles
        self.w, self.c1, self.c2 = w, c1, c2
        self._rng = np.random.RandomState(seed)
        self._independent = RandomSampler(seed=seed)
        self._reset()

    def _reset(self):
        self._names = None          # locked param order (sorted)
        self._dim = None
        self._pos = self._vel = None
        self._pbest_pos = self._pbest_val = None
        self._gbest_pos = None
        self._gbest_val = np.inf
        self._queue = []            # particle indices still to emit this gen
        self._pending = []          # (trial_number, particle_index) this gen

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False))

    def _init_swarm(self, dim):
        self._dim = dim
        self._pos = self._rng.rand(self.n_particles, dim)
        self._vel = self._rng.uniform(-0.1, 0.1, size=(self.n_particles, dim))
        self._pbest_pos = self._pos.copy()
        self._pbest_val = np.full(self.n_particles, np.inf)
        self._gbest_pos = self._pos[0].copy()
        self._gbest_val = np.inf
        self._queue = list(range(self.n_particles))
        self._pending = []

    def _end_generation(self, study):
        """Read this generation's trial values and update the swarm."""
        minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
        values = {t.number: t.value for t in study.get_trials(deepcopy=False)
                  if t.state == TrialState.COMPLETE and t.value is not None}
        for trial_number, i in self._pending:
            if trial_number not in values:
                continue  # pruned/failed trial: keep this particle's pbest
            fitness = values[trial_number] if minimize else -values[trial_number]
            if fitness < self._pbest_val[i]:
                self._pbest_val[i] = fitness
                self._pbest_pos[i] = self._pos[i].copy()
            if fitness < self._gbest_val:
                self._gbest_val = fitness
                self._gbest_pos = self._pos[i].copy()
        # PSO update toward personal and global bests.
        r1 = self._rng.rand(self.n_particles, self._dim)
        r2 = self._rng.rand(self.n_particles, self._dim)
        self._vel = (self.w * self._vel
                     + self.c1 * r1 * (self._pbest_pos - self._pos)
                     + self.c2 * r2 * (self._gbest_pos - self._pos))
        self._pos = np.clip(self._pos + self._vel, 0.0, 1.0)
        self._queue = list(range(self.n_particles))
        self._pending = []

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        names = sorted(search_space)
        if self._names != names:          # (re)lock the space and (re)init swarm
            self._reset()
            self._names = names
            self._init_swarm(len(names))
        if not self._queue:               # a full generation has been evaluated
            self._end_generation(study)
        i = self._queue.pop(0)
        self._pending.append((trial.number, i))
        return {name: _decode(self._pos[i][d], search_space[name])
                for d, name in enumerate(names)}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Fallback for the bootstrap trial (before the search space is known)
        # and for any dynamic parameter outside the relative space.
        return self._independent.sample_independent(
            study, trial, param_name, param_distribution)
