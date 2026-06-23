//! Swarm data structures.

/// A particle. All positions/velocities are internal `Vec<f64>`
/// (see the note in `traits.rs`).
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position (internal continuous representation).
    pub position: Vec<f64>,
    /// Current velocity.
    pub velocity: Vec<f64>,
    /// Best position visited by this particle (its `pbest`).
    pub best_position: Vec<f64>,
    /// Objective function value at `best_position`.
    pub best_value: f64,
}

impl Particle {
    /// Creates a particle, initializing its `pbest` with the given position.
    pub fn new(position: Vec<f64>, velocity: Vec<f64>, value: f64) -> Self {
        Self {
            best_position: position.clone(),
            best_value: value,
            position,
            velocity,
        }
    }
}

/// The complete swarm. The `Topology` receives it to decide neighborhoods.
#[derive(Debug, Clone)]
pub struct Swarm {
    /// The particles that make up the swarm.
    pub particles: Vec<Particle>,
}

impl Swarm {
    /// Creates a swarm from its particles.
    pub fn new(particles: Vec<Particle>) -> Self {
        Self { particles }
    }

    /// Number of particles in the swarm.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// `true` if the swarm has no particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}
