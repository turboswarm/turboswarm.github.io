//! The PSO optimizer. The loop knows nothing about any concrete variant:
//! it receives a `SearchSpace`, a `Velocity` and a `Topology` via generics.

use std::time::Instant;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::history::History;
use crate::params::PsoParams;
use crate::swarm::{Particle, Swarm};
use crate::traits::{best_among, SearchSpace, Topology, UpdateContext, Velocity};

/// Why the optimization loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ran the full `max_iterations`.
    MaxIterations,
    /// The global best reached the `target` value.
    Target,
    /// The evaluation budget (`max_evals`) was exhausted.
    MaxEvaluations,
    /// The global best stagnated for `patience` iterations.
    Stagnation,
    /// The wall-clock budget (`max_time`) elapsed.
    MaxTime,
    /// A per-iteration callback requested an early stop.
    Callback,
}

impl StopReason {
    /// A stable lowercase label, handy for the Python binding and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::MaxIterations => "max_iterations",
            StopReason::Target => "target",
            StopReason::MaxEvaluations => "max_evaluations",
            StopReason::Stagnation => "stagnation",
            StopReason::MaxTime => "max_time",
            StopReason::Callback => "callback",
        }
    }
}

/// Snapshot passed to a per-iteration callback. The callback returns `true`
/// to continue or `false` to stop early.
#[derive(Debug, Clone, Copy)]
pub struct IterationInfo<'a> {
    /// Current iteration index (0-based).
    pub iteration: usize,
    /// Best objective value found so far.
    pub best_value: f64,
    /// Best (internal, continuous) position found so far.
    pub best_position: &'a [f64],
    /// Objective evaluations performed so far.
    pub evaluations: usize,
}

/// Result of an optimization.
#[derive(Debug, Clone)]
pub struct PsoResult<T> {
    /// Best position found, already decoded to the space's type.
    pub best_position: Vec<T>,
    /// Best objective function value.
    pub best_value: f64,
    /// History of the run (empty if `record_history == false`).
    pub history: History,
    /// Total number of objective evaluations performed.
    pub evaluations: usize,
    /// Why the loop stopped.
    pub stop_reason: StopReason,
}

/// The optimizer. Generic over space, velocity rule and topology.
pub struct Pso<S, V, T>
where
    S: SearchSpace,
    V: Velocity,
    T: Topology,
{
    space: S,
    velocity: V,
    topology: T,
    params: PsoParams,
}

impl<S, V, T> Pso<S, V, T>
where
    S: SearchSpace,
    V: Velocity,
    T: Topology,
{
    /// Creates an optimizer from the space, the velocity rule, the
    /// topology and the parameters.
    pub fn new(space: S, velocity: V, topology: T, params: PsoParams) -> Self {
        Self {
            space,
            velocity,
            topology,
            params,
        }
    }

    /// Runs the optimization, minimizing `objective`.
    ///
    /// `objective` receives the already-decoded position (`S::Scalar`).
    pub fn minimize<F>(&self, objective: F) -> PsoResult<S::Scalar>
    where
        F: FnMut(&[S::Scalar]) -> f64,
    {
        self.run_sequential(objective, None)
    }

    /// Like [`minimize`](Self::minimize) but invokes `callback` once per
    /// iteration with an [`IterationInfo`] snapshot. The callback returns
    /// `true` to continue or `false` to stop early (reported as
    /// [`StopReason::Callback`]). Useful for live logging, custom stopping or
    /// progress reporting.
    pub fn minimize_with_callback<F, C>(
        &self,
        objective: F,
        mut callback: C,
    ) -> PsoResult<S::Scalar>
    where
        F: FnMut(&[S::Scalar]) -> f64,
        C: FnMut(&IterationInfo) -> bool,
    {
        self.run_sequential(objective, Some(&mut callback))
    }

    /// Shared sequential loop behind [`minimize`](Self::minimize) and
    /// [`minimize_with_callback`](Self::minimize_with_callback).
    fn run_sequential<F>(
        &self,
        mut objective: F,
        mut callback: Option<&mut dyn FnMut(&IterationInfo) -> bool>,
    ) -> PsoResult<S::Scalar>
    where
        F: FnMut(&[S::Scalar]) -> f64,
    {
        let start = Instant::now();
        let mut evaluations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        let mut rng: Box<dyn RngCore> = match self.params.seed {
            Some(s) => Box::new(ChaCha8Rng::seed_from_u64(s)),
            None => Box::new(ChaCha8Rng::from_entropy()),
        };

        // --- Swarm initialization ---
        let mut particles = Vec::with_capacity(self.params.n_particles);
        for _ in 0..self.params.n_particles {
            let pos = self.space.sample(rng.as_mut());
            let vel = self.space.sample_velocity(rng.as_mut());
            let value = objective(&self.space.decode(&pos));
            evaluations += 1;
            particles.push(Particle::new(pos, vel, value));
        }
        let mut swarm = Swarm::new(particles);

        let mut history = History::new();
        let (mut gbest_pos, mut gbest_val) = self.find_global_best(&swarm);

        // Early-stopping state (only used when `patience > 0`).
        let mut prev_best = gbest_val;
        let mut stagnation = 0usize;

        // Does the velocity rule need every neighbor's pbest (FIPS), or just
        // the best (inertia, constriction)? Deciding once avoids cloning the
        // whole neighborhood on the common path.
        let needs_neighborhood = self.velocity.needs_full_neighborhood();

        // Scratch buffers reused across particles and iterations to avoid a
        // per-particle heap allocation.
        let mut neighbor_best: Vec<f64> = Vec::new();
        let mut neighbor_best_data: Vec<Vec<f64>> = Vec::new();

        // --- Main loop ---
        for iter in 0..self.params.max_iterations {
            // Snapshot BEFORE moving (for animation of the iteration).
            if self.params.record_history {
                history
                    .positions
                    .push(swarm.particles.iter().map(|p| p.position.clone()).collect());
            }

            let n = swarm.len();
            for i in 0..n {
                // The neighborhood is computed ONCE; from it we derive the best
                // (classic variants) and, only when needed, the full list of
                // pbest (fully informed variants, e.g. FIPS).
                let neighbor_idx = self.topology.neighbors(i, &swarm);
                let nbest_idx = best_among(&swarm, &neighbor_idx, i);
                neighbor_best.clear();
                neighbor_best.extend_from_slice(&swarm.particles[nbest_idx].best_position);

                neighbor_best_data.clear();
                if needs_neighborhood {
                    for &k in &neighbor_idx {
                        neighbor_best_data.push(swarm.particles[k].best_position.clone());
                    }
                }
                let neighbor_bests: Vec<&[f64]> =
                    neighbor_best_data.iter().map(|v| v.as_slice()).collect();

                let new_vel = {
                    let p = &swarm.particles[i];
                    let ctx = UpdateContext {
                        position: &p.position,
                        velocity: &p.velocity,
                        personal_best: &p.best_position,
                        neighbor_best: &neighbor_best,
                        neighbor_bests: &neighbor_bests,
                        iteration: iter,
                        max_iterations: self.params.max_iterations,
                    };
                    self.velocity.update(&ctx, rng.as_mut())
                };

                let p = &mut swarm.particles[i];
                // Optional velocity clamp to [-v_max, v_max] per component.
                let mut new_vel = new_vel;
                if let Some(vmax) = self.params.v_max {
                    for v in new_vel.iter_mut() {
                        *v = v.clamp(-vmax, vmax);
                    }
                }
                for (pos, dv) in p.position.iter_mut().zip(&new_vel) {
                    *pos += dv;
                }
                p.velocity = new_vel;
                self.space.enforce_bounds(
                    &mut p.position,
                    &mut p.velocity,
                    self.params.bounds_handling,
                    rng.as_mut(),
                );

                let value = objective(&self.space.decode(&p.position));
                evaluations += 1;
                if value < p.best_value {
                    p.best_value = value;
                    // Reuse the existing allocation instead of cloning.
                    p.best_position.copy_from_slice(&p.position);
                    if value < gbest_val {
                        gbest_val = value;
                        gbest_pos.copy_from_slice(&p.position);
                    }
                }
            }

            if self.params.record_history {
                history.best_value.push(gbest_val);
                history.best_position.push(gbest_pos.clone());
            }

            // Per-iteration callback (may request an early stop).
            if let Some(cb) = callback.as_deref_mut() {
                let info = IterationInfo {
                    iteration: iter,
                    best_value: gbest_val,
                    best_position: &gbest_pos,
                    evaluations,
                };
                if !cb(&info) {
                    stop_reason = StopReason::Callback;
                    break;
                }
            }

            // Stop conditions. Stagnation needs mutable state, so it is handled
            // here; the budget conditions live in `budget_stop`.
            if self.params.patience > 0 {
                if prev_best - gbest_val > self.params.tol {
                    stagnation = 0;
                } else {
                    stagnation += 1;
                }
                prev_best = gbest_val;
                if stagnation >= self.params.patience {
                    stop_reason = StopReason::Stagnation;
                    break;
                }
            }
            if let Some(reason) = self.budget_stop(gbest_val, evaluations, start) {
                stop_reason = reason;
                break;
            }
        }

        PsoResult {
            best_position: self.space.decode(&gbest_pos),
            best_value: gbest_val,
            evaluations,
            stop_reason,
            history,
        }
    }

    /// Like [`minimize`](Self::minimize) but evaluates the swarm's objective in
    /// parallel with `rayon`. Worth it only when the objective is **expensive**;
    /// for cheap functions the parallelism overhead dominates.
    ///
    /// Note the **semantic difference**: this uses *synchronous* (Jacobi)
    /// updates — every particle moves using the previous iteration's bests, then
    /// the whole swarm is evaluated at once — whereas [`minimize`](Self::minimize)
    /// updates bests *asynchronously* within the iteration. Results therefore
    /// differ between the two (both are valid PSO schemes). The RNG is still
    /// drawn sequentially, so a fixed `seed` is fully reproducible.
    ///
    /// Requires the objective to be `Fn + Sync` (no per-evaluation mutable
    /// state); this is why it is not used for the Python-callable path, which is
    /// serialized by the GIL.
    pub fn minimize_parallel<F>(&self, objective: F) -> PsoResult<S::Scalar>
    where
        F: Fn(&[S::Scalar]) -> f64 + Sync,
        S: Sync,
        S::Scalar: Send,
    {
        // Capture only the space (not all of `self`), so the parallel closures
        // need `S: Sync` rather than `V: Sync`/`T: Sync`.
        let space = &self.space;
        let objective = &objective;
        self.run_synchronous(|swarm| {
            swarm
                .particles
                .par_iter()
                .map(|p| objective(&space.decode(&p.position)))
                .collect()
        })
    }

    /// Like [`minimize`](Self::minimize) but evaluates the WHOLE swarm in a
    /// single call to a batched objective `f(positions) -> values`, where
    /// `positions` are the decoded positions of every particle and the returned
    /// vector holds one value per particle (in the same order).
    ///
    /// This lets a vectorized objective (e.g. one written with NumPy from
    /// Python) amortize its overhead over the swarm instead of paying it per
    /// particle. Like [`minimize_parallel`](Self::minimize_parallel), it uses
    /// *synchronous* updates.
    pub fn minimize_batch<F>(&self, mut objective: F) -> PsoResult<S::Scalar>
    where
        F: FnMut(&[Vec<S::Scalar>]) -> Vec<f64>,
    {
        let space = &self.space;
        self.run_synchronous(|swarm| {
            let decoded: Vec<Vec<S::Scalar>> = swarm
                .particles
                .iter()
                .map(|p| space.decode(&p.position))
                .collect();
            objective(&decoded)
        })
    }

    /// Shared synchronous (Jacobi) loop: move the whole swarm, evaluate it in
    /// one batch via `evaluate`, then update the bests. Backs
    /// [`minimize_parallel`](Self::minimize_parallel) and
    /// [`minimize_batch`](Self::minimize_batch). `evaluate` must return one
    /// value per particle, in order.
    fn run_synchronous<E>(&self, mut evaluate: E) -> PsoResult<S::Scalar>
    where
        E: FnMut(&Swarm) -> Vec<f64>,
    {
        let start = Instant::now();
        let mut evaluations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        let mut rng: Box<dyn RngCore> = match self.params.seed {
            Some(s) => Box::new(ChaCha8Rng::seed_from_u64(s)),
            None => Box::new(ChaCha8Rng::from_entropy()),
        };

        // --- Swarm initialization (sampling sequential, evaluation batched) ---
        let mut particles = Vec::with_capacity(self.params.n_particles);
        for _ in 0..self.params.n_particles {
            let pos = self.space.sample(rng.as_mut());
            let vel = self.space.sample_velocity(rng.as_mut());
            particles.push(Particle::new(pos, vel, f64::INFINITY));
        }
        let mut swarm = Swarm::new(particles);
        let init_vals = evaluate(&swarm);
        evaluations += init_vals.len();
        for (p, v) in swarm.particles.iter_mut().zip(init_vals) {
            p.best_value = v;
        }

        let mut history = History::new();
        let (mut gbest_pos, mut gbest_val) = self.find_global_best(&swarm);
        let mut prev_best = gbest_val;
        let mut stagnation = 0usize;

        let needs_neighborhood = self.velocity.needs_full_neighborhood();
        let mut neighbor_best: Vec<f64> = Vec::new();
        let mut neighbor_best_data: Vec<Vec<f64>> = Vec::new();

        for iter in 0..self.params.max_iterations {
            if self.params.record_history {
                history
                    .positions
                    .push(swarm.particles.iter().map(|p| p.position.clone()).collect());
            }

            let n = swarm.len();

            // Phase 1 (sequential, consumes the RNG): move every particle using
            // the bests from the start of the iteration.
            for i in 0..n {
                let neighbor_idx = self.topology.neighbors(i, &swarm);
                let nbest_idx = best_among(&swarm, &neighbor_idx, i);
                neighbor_best.clear();
                neighbor_best.extend_from_slice(&swarm.particles[nbest_idx].best_position);

                neighbor_best_data.clear();
                if needs_neighborhood {
                    for &k in &neighbor_idx {
                        neighbor_best_data.push(swarm.particles[k].best_position.clone());
                    }
                }
                let neighbor_bests: Vec<&[f64]> =
                    neighbor_best_data.iter().map(|v| v.as_slice()).collect();

                let new_vel = {
                    let p = &swarm.particles[i];
                    let ctx = UpdateContext {
                        position: &p.position,
                        velocity: &p.velocity,
                        personal_best: &p.best_position,
                        neighbor_best: &neighbor_best,
                        neighbor_bests: &neighbor_bests,
                        iteration: iter,
                        max_iterations: self.params.max_iterations,
                    };
                    self.velocity.update(&ctx, rng.as_mut())
                };

                let p = &mut swarm.particles[i];
                let mut new_vel = new_vel;
                if let Some(vmax) = self.params.v_max {
                    for v in new_vel.iter_mut() {
                        *v = v.clamp(-vmax, vmax);
                    }
                }
                for (pos, dv) in p.position.iter_mut().zip(&new_vel) {
                    *pos += dv;
                }
                p.velocity = new_vel;
                self.space.enforce_bounds(
                    &mut p.position,
                    &mut p.velocity,
                    self.params.bounds_handling,
                    rng.as_mut(),
                );
            }

            // Phase 2: evaluate the whole swarm at once.
            let values = evaluate(&swarm);
            assert_eq!(
                values.len(),
                swarm.len(),
                "batched objective must return one value per particle"
            );
            evaluations += values.len();

            // Phase 3 (sequential): update personal and global bests.
            for (i, &value) in values.iter().enumerate() {
                let p = &mut swarm.particles[i];
                if value < p.best_value {
                    p.best_value = value;
                    p.best_position.copy_from_slice(&p.position);
                    if value < gbest_val {
                        gbest_val = value;
                        gbest_pos.copy_from_slice(&p.position);
                    }
                }
            }

            if self.params.record_history {
                history.best_value.push(gbest_val);
                history.best_position.push(gbest_pos.clone());
            }

            if self.params.patience > 0 {
                if prev_best - gbest_val > self.params.tol {
                    stagnation = 0;
                } else {
                    stagnation += 1;
                }
                prev_best = gbest_val;
                if stagnation >= self.params.patience {
                    stop_reason = StopReason::Stagnation;
                    break;
                }
            }
            if let Some(reason) = self.budget_stop(gbest_val, evaluations, start) {
                stop_reason = reason;
                break;
            }
        }

        PsoResult {
            best_position: self.space.decode(&gbest_pos),
            best_value: gbest_val,
            evaluations,
            stop_reason,
            history,
        }
    }

    fn find_global_best(&self, swarm: &Swarm) -> (Vec<f64>, f64) {
        let best = swarm
            .particles
            .iter()
            .min_by(|a, b| a.best_value.partial_cmp(&b.best_value).unwrap())
            .expect("swarm not empty");
        (best.best_position.clone(), best.best_value)
    }

    /// Checks the budget-based stop conditions (target value, evaluation
    /// budget, wall-clock budget). Returns the reason to stop, if any. The
    /// stagnation criterion is handled by the caller (it needs mutable state).
    fn budget_stop(
        &self,
        gbest_val: f64,
        evaluations: usize,
        start: Instant,
    ) -> Option<StopReason> {
        if let Some(t) = self.params.target {
            if gbest_val <= t {
                return Some(StopReason::Target);
            }
        }
        if let Some(m) = self.params.max_evals {
            if evaluations >= m {
                return Some(StopReason::MaxEvaluations);
            }
        }
        if let Some(d) = self.params.max_time {
            if start.elapsed() >= d {
                return Some(StopReason::MaxTime);
            }
        }
        None
    }
}
