//! Core traits of the framework.
//!
//! Design philosophy: the PSO loop (in `pso.rs`) knows nothing about any
//! concrete variant. Everything that changes between variants lives behind
//! these three traits. Adding a new variant = implementing a trait,
//! without touching the core.
//!
//! Internal convention: ALL positions are represented as `Vec<f64>`
//! within the optimizer. The integer/real difference lives solely in
//! `SearchSpace::decode`, which translates the internal continuous
//! representation into the actual type the objective function is evaluated on.

use rand::RngCore;

use crate::swarm::Swarm;

/// What to do with a particle that leaves the feasible region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryHandling {
    /// Clip the position to the boundary (the classic default).
    Clamp,
    /// Reflect ("bounce") the position back into range and flip the offending
    /// velocity component.
    Reflect,
    /// Wrap around toroidally to the opposite side of the range.
    Wrap,
    /// Re-sample the offending component uniformly within bounds and zero its
    /// velocity.
    Reinit,
}

/// Defines the problem domain: dimension, bounds, initial sampling,
/// clamping to the feasible region and decoding to the evaluable type.
pub trait SearchSpace {
    /// Scalar type the objective function is evaluated on
    /// (`f64` for reals, `i64` for integers).
    type Scalar: Clone + Copy + std::fmt::Debug;

    /// Number of dimensions of the problem.
    fn dim(&self) -> usize;

    /// Generates a random initial position within the bounds.
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64>;

    /// Generates an initial velocity (by default, proportional to the range).
    fn sample_velocity(&self, rng: &mut dyn RngCore) -> Vec<f64>;

    /// Clamps a position to the feasible region (the bounds).
    fn clamp(&self, position: &mut [f64]);

    /// Enforces the bounds on a particle using the chosen strategy, possibly
    /// adjusting its velocity (and drawing from `rng` for `Reinit`).
    ///
    /// The default implementation ignores `handling` and simply clamps, so
    /// custom spaces keep working; the built-in spaces override it to support
    /// all strategies.
    fn enforce_bounds(
        &self,
        position: &mut [f64],
        _velocity: &mut [f64],
        _handling: BoundaryHandling,
        _rng: &mut dyn RngCore,
    ) {
        self.clamp(position);
    }

    /// Translates the internal continuous position to the evaluable type.
    /// For reals it is the identity; for integers it applies the discretization.
    fn decode(&self, raw: &[f64]) -> Vec<Self::Scalar>;

    /// Per-dimension `(min, max)` of the internal continuous domain, as floats.
    /// Used by operators that need the domain extent (e.g. the MOPSO mutation).
    /// The default returns an empty vector, which simply disables those
    /// operators; the built-in spaces override it.
    fn span(&self) -> Vec<(f64, f64)> {
        Vec::new()
    }
}

/// Context passed to a velocity update rule.
/// Contains everything any PSO variant might need.
pub struct UpdateContext<'a> {
    /// Current position of the particle.
    pub position: &'a [f64],
    /// Current velocity of the particle.
    pub velocity: &'a [f64],
    /// Personal best position of the particle (`pbest`).
    pub personal_best: &'a [f64],
    /// Neighborhood best (determined by the `Topology`).
    pub neighbor_best: &'a [f64],
    /// `personal_best` of ALL neighbors in the neighborhood (including the
    /// particle itself). Used by *fully informed* variants like FIPS, which
    /// are influenced by the whole neighborhood and not just by its best. The
    /// classic variants (inertia, constriction) ignore it.
    pub neighbor_bests: &'a [&'a [f64]],
    /// Current iteration (0-based), useful for decay schemes.
    pub iteration: usize,
    /// Total number of iterations of the run.
    pub max_iterations: usize,
}

/// The velocity update rule: variants plug in HERE.
///
/// Examples: inertia weight, constriction factor (Clerc-Kennedy),
/// FIPS, etc. To add a new variant, implement only this trait.
pub trait Velocity {
    /// Returns the NEW velocity for a particle.
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64>;

    /// Whether this rule reads the whole neighborhood
    /// ([`UpdateContext::neighbor_bests`]). Defaults to `false`; fully informed
    /// variants (FIPS) override it to `true`. The loop only gathers the
    /// neighborhood's `pbest` list when this returns `true`, so the classic
    /// variants pay no cloning cost for information they ignore.
    fn needs_full_neighborhood(&self) -> bool {
        false
    }
}

/// The social structure of the swarm: who informs whom.
///
/// Examples: `GlobalBest` (everyone sees the global best), ring, Von Neumann.
///
/// The fundamental method is [`neighbors`](Topology::neighbors): it defines the
/// neighborhood. [`best_neighbor`](Topology::best_neighbor) is derived from it
/// by default, so a new topology only needs to implement `neighbors`.
pub trait Topology {
    /// The neighborhood of particle `i`: the indices of the particles whose
    /// `personal_best` informs it. By convention it INCLUDES `i` itself.
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize>;

    /// Index of the neighbor with the best `personal_best`. Derived from
    /// `neighbors` by default; it is what the variants that only look at the
    /// best use (inertia, constriction).
    fn best_neighbor(&self, i: usize, swarm: &Swarm) -> usize {
        let nb = self.neighbors(i, swarm);
        best_among(swarm, &nb, i)
    }
}

// --- Dynamic dispatch ---
// The core uses generics (zero-cost) in Rust. But at the Python boundary
// the variants are chosen at runtime, so we need `Box<dyn ...>`. These impls
// allow using a Box as if it were the trait, so that
// `Pso<S, Box<dyn Velocity>, Box<dyn Topology>>` compiles.

impl Velocity for Box<dyn Velocity> {
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64> {
        (**self).update(ctx, rng)
    }
    fn needs_full_neighborhood(&self) -> bool {
        (**self).needs_full_neighborhood()
    }
}

impl Topology for Box<dyn Topology> {
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize> {
        (**self).neighbors(i, swarm)
    }
    fn best_neighbor(&self, i: usize, swarm: &Swarm) -> usize {
        (**self).best_neighbor(i, swarm)
    }
}

/// Returns the index (among `candidates`) with the best (smallest)
/// `personal_best`. On a tie it returns the FIRST one in `candidates`
/// (important for determinism). If `candidates` is empty, `fallback`.
pub fn best_among(swarm: &Swarm, candidates: &[usize], fallback: usize) -> usize {
    candidates
        .iter()
        .copied()
        .min_by(|&a, &b| {
            swarm.particles[a]
                .best_value
                .partial_cmp(&swarm.particles[b].best_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(fallback)
}

/// Convenience: the global best particle of the swarm.
pub fn global_best_index(swarm: &Swarm) -> usize {
    best_among(swarm, &(0..swarm.len()).collect::<Vec<_>>(), 0)
}
