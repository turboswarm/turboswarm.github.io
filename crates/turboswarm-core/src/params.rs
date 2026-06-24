//! Optimizer configuration parameters.

/// Parameters common to the PSO variants.
///
/// Note: `w`, `c1`, `c2` are used by the concrete velocity rules
/// (see `velocity/`). Not all variants use all three.
#[derive(Debug, Clone)]
pub struct PsoParams {
    /// Inertia weight (used by the inertia variant).
    pub w: f64,
    /// Cognitive coefficient (attraction to the personal best).
    pub c1: f64,
    /// Social coefficient (attraction to the neighborhood best).
    pub c2: f64,
    /// Number of particles in the swarm.
    pub n_particles: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// RNG seed. `None` => random system seed.
    /// Fix it for reproducible experiments.
    pub seed: Option<u64>,
    /// If `true`, stores the positions of all particles at each
    /// iteration (needed for animation; memory-expensive).
    pub record_history: bool,
    /// Optional velocity clamp: if `Some(vmax)`, every velocity component is
    /// clamped to `[-vmax, vmax]` after each update. `None` disables it.
    pub v_max: Option<f64>,
    /// Early-stopping window: stop when the global best does not improve by
    /// more than `tol` for `patience` consecutive iterations. `0` disables
    /// early stopping (the run always uses all `max_iterations`).
    pub patience: usize,
    /// Minimum improvement in the global best value that counts as progress
    /// for the `patience` window. Only used when `patience > 0`.
    pub tol: f64,
    /// Optional budget: stop once this many objective evaluations have been
    /// made. `None` disables it.
    pub max_evals: Option<usize>,
    /// Optional target: stop as soon as the global best value is `<= target`.
    /// `None` disables it.
    pub target: Option<f64>,
    /// Optional wall-clock budget: stop once this much time has elapsed.
    /// `None` disables it.
    pub max_time: Option<std::time::Duration>,
    /// How to handle particles that leave the bounds. Defaults to `Clamp`.
    pub bounds_handling: crate::traits::BoundaryHandling,
}

impl Default for PsoParams {
    fn default() -> Self {
        // Classic values (Shi & Eberhart), reasonable as defaults.
        Self {
            w: 0.729,
            c1: 1.49445,
            c2: 1.49445,
            n_particles: 30,
            max_iterations: 100,
            seed: None,
            record_history: true,
            v_max: None,
            patience: 0,
            tol: 0.0,
            max_evals: None,
            target: None,
            max_time: None,
            bounds_handling: crate::traits::BoundaryHandling::Clamp,
        }
    }
}
