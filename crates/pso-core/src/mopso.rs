//! Multi-objective PSO (MOPSO, Coello Coello & Lechuga, 2004).
//!
//! Unlike the single-objective optimizer, there is no single best: the result
//! is an approximation of the **Pareto front** — the set of non-dominated
//! trade-off solutions. This module is independent of the single-objective core
//! but reuses [`SearchSpace`] and the [`Velocity`] trait (the archive *leader*
//! plays the role of the social attractor / `neighbor_best`).
//!
//! Use a single-leader velocity rule (inertia or constriction); FIPS, which
//! needs the whole neighborhood, does not apply here.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::traits::{SearchSpace, UpdateContext, Velocity};

/// Parameters for the multi-objective optimizer.
#[derive(Debug, Clone)]
pub struct MopsoParams {
    /// Number of particles.
    pub n_particles: usize,
    /// Number of iterations.
    pub max_iterations: usize,
    /// Maximum size of the external archive (the returned front is pruned to
    /// this by crowding distance).
    pub archive_size: usize,
    /// RNG seed; fix it for reproducible runs.
    pub seed: Option<u64>,
    /// Turbulence/mutation strength in `[0, 1]`. Each iteration a particle is
    /// mutated with probability `mutation_rate · (1 − iter/max_iter)`, within a
    /// window that shrinks over time. `0` disables it. Improves front spread
    /// and helps escape local fronts.
    pub mutation_rate: f64,
}

impl Default for MopsoParams {
    fn default() -> Self {
        Self {
            n_particles: 100,
            max_iterations: 100,
            archive_size: 100,
            seed: None,
            mutation_rate: 0.1,
        }
    }
}

/// A single non-dominated solution of the Pareto front.
#[derive(Debug, Clone)]
pub struct MoSolution<T> {
    /// Decision variables, decoded to the space's type.
    pub position: Vec<T>,
    /// Objective values at `position`.
    pub objectives: Vec<f64>,
}

/// Result of a multi-objective run: the approximated Pareto front.
#[derive(Debug, Clone)]
pub struct MopsoResult<T> {
    /// The non-dominated solutions found.
    pub front: Vec<MoSolution<T>>,
}

/// `true` if `a` Pareto-dominates `b` (minimization): no worse in every
/// objective and strictly better in at least one.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }
        if x < y {
            strictly_better = true;
        }
    }
    strictly_better
}

/// NSGA-II crowding distance for a set of objective vectors (larger = more
/// isolated, hence more valuable for diversity).
fn crowding_distance(objs: &[Vec<f64>]) -> Vec<f64> {
    let n = objs.len();
    let mut dist = vec![0.0_f64; n];
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }
    let m = objs[0].len();
    #[allow(clippy::needless_range_loop)]
    for k in 0..m {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            objs[a][k]
                .partial_cmp(&objs[b][k])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        dist[idx[0]] = f64::INFINITY;
        dist[idx[n - 1]] = f64::INFINITY;
        let range = objs[idx[n - 1]][k] - objs[idx[0]][k];
        if range > 0.0 {
            for j in 1..n - 1 {
                dist[idx[j]] += (objs[idx[j + 1]][k] - objs[idx[j - 1]][k]) / range;
            }
        }
    }
    dist
}

/// External archive of non-dominated solutions (raw internal positions).
struct Archive {
    members: Vec<(Vec<f64>, Vec<f64>)>, // (raw position, objectives)
    max_size: usize,
}

impl Archive {
    fn new(max_size: usize) -> Self {
        Self {
            members: Vec::new(),
            max_size,
        }
    }

    /// Inserts a candidate if it is not dominated, removing any members it
    /// dominates. Exact-objective duplicates are skipped.
    fn insert(&mut self, pos: &[f64], obj: &[f64]) {
        if self.members.iter().any(|(_, o)| dominates(o, obj)) {
            return;
        }
        self.members.retain(|(_, o)| !dominates(obj, o));
        if self.members.iter().any(|(_, o)| o == obj) {
            return;
        }
        self.members.push((pos.to_vec(), obj.to_vec()));
    }

    /// Prunes to `max_size` keeping the most isolated (largest crowding).
    /// Returns the crowding distances of the kept members, aligned by index.
    fn prune_and_crowding(&mut self) -> Vec<f64> {
        let objs: Vec<Vec<f64>> = self.members.iter().map(|(_, o)| o.clone()).collect();
        let cd = crowding_distance(&objs);
        if self.members.len() <= self.max_size {
            return cd;
        }
        let mut order: Vec<usize> = (0..self.members.len()).collect();
        order.sort_by(|&a, &b| {
            cd[b]
                .partial_cmp(&cd[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(self.max_size);
        order.sort_unstable();
        self.members = order.iter().map(|&i| self.members[i].clone()).collect();
        let objs: Vec<Vec<f64>> = self.members.iter().map(|(_, o)| o.clone()).collect();
        crowding_distance(&objs)
    }
}

/// The multi-objective optimizer. Generic over the search space and the
/// velocity rule (the topology is implicit: the leader comes from the archive).
pub struct Mopso<S, V>
where
    S: SearchSpace,
    V: Velocity,
{
    space: S,
    velocity: V,
    params: MopsoParams,
}

impl<S, V> Mopso<S, V>
where
    S: SearchSpace,
    V: Velocity,
{
    /// Creates a multi-objective optimizer.
    pub fn new(space: S, velocity: V, params: MopsoParams) -> Self {
        Self {
            space,
            velocity,
            params,
        }
    }

    /// Minimizes a vector-valued `objectives` function, returning the Pareto
    /// front. `objectives` receives the decoded position and returns one value
    /// per objective (all minimized).
    pub fn minimize<F>(&self, mut objectives: F) -> MopsoResult<S::Scalar>
    where
        F: FnMut(&[S::Scalar]) -> Vec<f64>,
    {
        let mut rng: Box<dyn RngCore> = match self.params.seed {
            Some(s) => Box::new(ChaCha8Rng::seed_from_u64(s)),
            None => Box::new(ChaCha8Rng::from_entropy()),
        };

        // --- Initialization ---
        let mut positions = Vec::with_capacity(self.params.n_particles);
        let mut velocities = Vec::with_capacity(self.params.n_particles);
        let mut objs = Vec::with_capacity(self.params.n_particles);
        for _ in 0..self.params.n_particles {
            let pos = self.space.sample(rng.as_mut());
            let vel = self.space.sample_velocity(rng.as_mut());
            let o = objectives(&self.space.decode(&pos));
            positions.push(pos);
            velocities.push(vel);
            objs.push(o);
        }
        // Personal bests start at the initial positions.
        let mut pbest_pos = positions.clone();
        let mut pbest_obj = objs.clone();

        let mut archive = Archive::new(self.params.archive_size);
        for (p, o) in positions.iter().zip(&objs) {
            archive.insert(p, o);
        }

        // Domain extent per dimension (for the mutation window); empty disables.
        let span = self.space.span();
        let max_iter = self.params.max_iterations.max(1) as f64;

        // --- Main loop ---
        for iter in 0..self.params.max_iterations {
            let crowding = archive.prune_and_crowding();
            // Snapshot the leader pool so it stays aligned with `crowding`
            // while the archive grows from this iteration's insertions.
            let leaders: Vec<Vec<f64>> = archive.members.iter().map(|(p, _)| p.clone()).collect();

            for i in 0..self.params.n_particles {
                // Leader: binary tournament favoring the less crowded region.
                let leader_pos = {
                    let a = rng.gen_range(0..leaders.len());
                    let b = rng.gen_range(0..leaders.len());
                    if crowding[a] >= crowding[b] {
                        leaders[a].clone()
                    } else {
                        leaders[b].clone()
                    }
                };

                let new_vel = {
                    let ctx = UpdateContext {
                        position: &positions[i],
                        velocity: &velocities[i],
                        personal_best: &pbest_pos[i],
                        neighbor_best: &leader_pos,
                        neighbor_bests: &[],
                        iteration: iter,
                        max_iterations: self.params.max_iterations,
                    };
                    self.velocity.update(&ctx, rng.as_mut())
                };

                for (x, dv) in positions[i].iter_mut().zip(&new_vel) {
                    *x += dv;
                }
                velocities[i] = new_vel;
                self.space.clamp(&mut positions[i]);

                // Turbulence/mutation: with a decreasing probability, perturb a
                // random dimension within a shrinking window around its value.
                if self.params.mutation_rate > 0.0 && !span.is_empty() {
                    let decay = 1.0 - iter as f64 / max_iter;
                    if rng.gen::<f64>() < self.params.mutation_rate * decay {
                        let d = rng.gen_range(0..span.len());
                        let (lo, hi) = span[d];
                        let half = (hi - lo) * decay * 0.5;
                        if half > 0.0 {
                            let nlo = (positions[i][d] - half).max(lo);
                            let nhi = (positions[i][d] + half).min(hi);
                            if nhi > nlo {
                                positions[i][d] = rng.gen_range(nlo..=nhi);
                            }
                        }
                    }
                }

                let new_obj = objectives(&self.space.decode(&positions[i]));

                // Personal-best update by Pareto dominance.
                if dominates(&new_obj, &pbest_obj[i]) {
                    pbest_pos[i] = positions[i].clone();
                    pbest_obj[i] = new_obj.clone();
                } else if !dominates(&pbest_obj[i], &new_obj) && rng.gen::<bool>() {
                    // Mutually non-dominated: keep one at random.
                    pbest_pos[i] = positions[i].clone();
                    pbest_obj[i] = new_obj.clone();
                }

                archive.insert(&positions[i], &new_obj);
            }
        }

        archive.prune_and_crowding();
        let front = archive
            .members
            .into_iter()
            .map(|(pos, objectives)| MoSolution {
                position: self.space.decode(&pos),
                objectives,
            })
            .collect();
        MopsoResult { front }
    }
}
