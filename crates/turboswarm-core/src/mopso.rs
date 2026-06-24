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

use std::collections::HashMap;

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
    /// this size).
    pub archive_size: usize,
    /// RNG seed; fix it for reproducible runs.
    pub seed: Option<u64>,
    /// Turbulence/mutation strength in `[0, 1]`. Each iteration a particle is
    /// mutated with probability `mutation_rate · (1 − iter/max_iter)`, within a
    /// window that shrinks over time. `0` disables it. Improves front spread
    /// and helps escape local fronts.
    pub mutation_rate: f64,
    /// Archive diversity strategy. `None` (default) keeps the most isolated
    /// members by **NSGA-II crowding distance**. `Some(d)` uses Coello's
    /// **adaptive hypercube grid** with `d` divisions per objective: pruning
    /// drops members from the most crowded cell, and leaders are drawn towards
    /// sparser cells. The grid is the mechanism from the original MOPSO paper;
    /// it tends to spread the front more evenly, especially with many members.
    pub grid_divisions: Option<usize>,
}

impl Default for MopsoParams {
    fn default() -> Self {
        Self {
            n_particles: 100,
            max_iterations: 100,
            archive_size: 100,
            seed: None,
            mutation_rate: 0.1,
            grid_divisions: None,
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

impl<T> MopsoResult<T> {
    /// Hypervolume of this front (see [`hypervolume`]). If `reference` is
    /// `None`, a reference point is derived from the front's nadir (per
    /// objective, the worst value plus 10% of its spread), which is convenient
    /// for a single run but **not** comparable across runs — pass an explicit,
    /// shared `reference` when comparing fronts.
    pub fn hypervolume(&self, reference: Option<&[f64]>) -> f64 {
        let objs: Vec<Vec<f64>> = self.front.iter().map(|s| s.objectives.clone()).collect();
        match reference {
            Some(r) => hypervolume(&objs, r),
            None => hypervolume(&objs, &nadir_reference(&objs)),
        }
    }
}

/// A reference point just beyond the worst corner of `objs`: per objective, the
/// maximum value plus 10% of its observed spread (or `+1` for a flat axis).
/// Empty input yields an empty reference.
pub fn nadir_reference(objs: &[Vec<f64>]) -> Vec<f64> {
    if objs.is_empty() {
        return Vec::new();
    }
    let m = objs[0].len();
    (0..m)
        .map(|k| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for o in objs {
                lo = lo.min(o[k]);
                hi = hi.max(o[k]);
            }
            let span = hi - lo;
            hi + if span > 0.0 { 0.1 * span } else { 1.0 }
        })
        .collect()
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

/// Non-dominated subset of `pts` (minimization), deduplicated.
fn non_dominated(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        let dominated = pts
            .iter()
            .enumerate()
            .any(|(j, q)| j != i && dominates(q, p));
        if !dominated && !out.iter().any(|o| o == p) {
            out.push(p.clone());
        }
    }
    out
}

/// Hypervolume indicator: the volume of objective space that is dominated by
/// `front` and bounded above by `reference` (minimization — larger is better).
///
/// It is the standard quality metric for comparing Pareto-front approximations,
/// rewarding both convergence (closeness to the true front) and spread, with no
/// reference front required. Uses the WFG algorithm (While et al., 2012), exact
/// for any number of objectives.
///
/// `reference` must be a point each counted solution beats in every objective
/// (component-wise larger than the front). Points not strictly better than
/// `reference` in all objectives contribute nothing. To compare two fronts the
/// reference point must be identical for both.
///
/// # Example
/// ```
/// use turboswarm_core::mopso::hypervolume;
/// // The staircase (1,3), (2,2), (3,1) under reference (4,4) covers area 6.
/// let front = [vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
/// let hv = hypervolume(&front, &[4.0, 4.0]);
/// assert!((hv - 6.0).abs() < 1e-9);
/// ```
pub fn hypervolume(front: &[Vec<f64>], reference: &[f64]) -> f64 {
    let pts: Vec<Vec<f64>> = front
        .iter()
        .filter(|p| p.len() == reference.len() && p.iter().zip(reference).all(|(x, r)| x < r))
        .cloned()
        .collect();
    wfg(&non_dominated(&pts), reference)
}

/// WFG hypervolume of a non-dominated point set: the sum of each point's
/// *exclusive* contribution. Order-independent.
fn wfg(pl: &[Vec<f64>], reference: &[f64]) -> f64 {
    (0..pl.len()).map(|k| exclhv(pl, k, reference)).sum()
}

/// Exclusive hypervolume of `pl[k]`: its full box minus the part already
/// covered by the points that follow it (limited from below by `pl[k]`).
fn exclhv(pl: &[Vec<f64>], k: usize, reference: &[f64]) -> f64 {
    let incl: f64 = pl[k]
        .iter()
        .zip(reference)
        .map(|(x, r)| (r - x).max(0.0))
        .product();
    incl - wfg(&non_dominated(&limit_set(pl, k)), reference)
}

/// The points after `k`, each pushed away from the origin to the worse
/// (component-wise max, for minimization) of itself and `pl[k]`.
fn limit_set(pl: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    pl[k + 1..]
        .iter()
        .map(|q| pl[k].iter().zip(q).map(|(a, b)| a.max(*b)).collect())
        .collect()
}

/// Locates each objective vector in an adaptive hypercube grid of `divisions`
/// cells per objective, sized to the current min/max of `objs` along each
/// objective. Returns the integer cell coordinate of every point. Coello's
/// adaptive grid: the bounds follow the archive, so the cells always cover it.
fn grid_cells(objs: &[Vec<f64>], divisions: usize) -> Vec<Vec<usize>> {
    if objs.is_empty() {
        return Vec::new();
    }
    let m = objs[0].len();
    let div = divisions.max(1);
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for o in objs {
        for k in 0..m {
            lo[k] = lo[k].min(o[k]);
            hi[k] = hi[k].max(o[k]);
        }
    }
    objs.iter()
        .map(|o| {
            (0..m)
                .map(|k| {
                    let span = hi[k] - lo[k];
                    if span <= 0.0 {
                        0
                    } else {
                        let idx = ((o[k] - lo[k]) / span * div as f64) as usize;
                        idx.min(div - 1)
                    }
                })
                .collect()
        })
        .collect()
}

/// Cell populations: how many members share each member's grid cell, aligned by
/// index.
fn cell_populations(cells: &[Vec<usize>]) -> Vec<usize> {
    let mut counts: HashMap<&Vec<usize>, usize> = HashMap::new();
    for c in cells {
        *counts.entry(c).or_insert(0) += 1;
    }
    cells.iter().map(|c| counts[c]).collect()
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
    /// `Some(d)` selects Coello's adaptive grid (with `d` divisions per
    /// objective); `None` uses crowding distance.
    grid_divisions: Option<usize>,
}

impl Archive {
    fn new(max_size: usize, grid_divisions: Option<usize>) -> Self {
        Self {
            members: Vec::new(),
            max_size,
            grid_divisions,
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

    /// Prunes to `max_size` and returns a per-member **sparsity score** (larger
    /// = more isolated, hence preferred as a leader), aligned by index. Uses the
    /// crowding distance or Coello's grid depending on `grid_divisions`.
    fn prune_and_scores(&mut self) -> Vec<f64> {
        match self.grid_divisions {
            None => self.prune_and_crowding(),
            Some(div) => self.prune_and_grid(div),
        }
    }

    /// Prunes keeping the largest crowding distance; returns those distances.
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

    /// Coello's adaptive-grid pruning: repeatedly drop a member from the most
    /// crowded hypercube until within `max_size`. Returns `1 / cell_population`
    /// per surviving member, so members in sparser cells score higher.
    fn prune_and_grid(&mut self, div: usize) -> Vec<f64> {
        while self.members.len() > self.max_size {
            let objs: Vec<Vec<f64>> = self.members.iter().map(|(_, o)| o.clone()).collect();
            let pops = cell_populations(&grid_cells(&objs, div));
            // Drop a member from the most populated cell (ties: the last one,
            // chosen deterministically to keep runs reproducible).
            let victim = (0..pops.len()).max_by_key(|&i| pops[i]).unwrap();
            self.members.remove(victim);
        }
        let objs: Vec<Vec<f64>> = self.members.iter().map(|(_, o)| o.clone()).collect();
        cell_populations(&grid_cells(&objs, div))
            .into_iter()
            .map(|p| 1.0 / p as f64)
            .collect()
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

        let mut archive = Archive::new(self.params.archive_size, self.params.grid_divisions);
        for (p, o) in positions.iter().zip(&objs) {
            archive.insert(p, o);
        }

        // Domain extent per dimension (for the mutation window); empty disables.
        let span = self.space.span();
        let max_iter = self.params.max_iterations.max(1) as f64;

        // --- Main loop ---
        for iter in 0..self.params.max_iterations {
            let scores = archive.prune_and_scores();
            // Snapshot the leader pool so it stays aligned with `scores`
            // while the archive grows from this iteration's insertions.
            let leaders: Vec<Vec<f64>> = archive.members.iter().map(|(p, _)| p.clone()).collect();

            for i in 0..self.params.n_particles {
                // Leader: binary tournament favoring the sparser region (higher
                // crowding distance, or sparser grid cell).
                let leader_pos = {
                    let a = rng.gen_range(0..leaders.len());
                    let b = rng.gen_range(0..leaders.len());
                    if scores[a] >= scores[b] {
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

        archive.prune_and_scores();
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
