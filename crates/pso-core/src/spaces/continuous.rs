//! Continuous search space (real variables). Classic PSO.

use rand::{Rng, RngCore};

use crate::traits::SearchSpace;

/// Continuous search box defined by `[min, max]` bounds per dimension.
#[derive(Debug, Clone)]
pub struct ContinuousSpace {
    bounds: Vec<(f64, f64)>,
}

impl ContinuousSpace {
    /// Creates the space from the per-dimension bounds.
    /// Panics if any bound has `min > max`.
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        for (i, (lo, hi)) in bounds.iter().enumerate() {
            assert!(lo <= hi, "invalid bound in dimension {i}: {lo} > {hi}");
        }
        Self { bounds }
    }

    /// The `(min, max)` bounds per dimension.
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }
}

impl SearchSpace for ContinuousSpace {
    type Scalar = f64;

    fn dim(&self) -> usize {
        self.bounds.len()
    }

    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo..=hi))
            .collect()
    }

    fn sample_velocity(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        // Initial velocity in [-range, +range], a common practice.
        self.bounds
            .iter()
            .map(|&(lo, hi)| {
                let range = hi - lo;
                rng.gen_range(-range..=range)
            })
            .collect()
    }

    fn clamp(&self, position: &mut [f64]) {
        for (x, &(lo, hi)) in position.iter_mut().zip(&self.bounds) {
            *x = x.clamp(lo, hi);
        }
    }

    fn enforce_bounds(
        &self,
        position: &mut [f64],
        velocity: &mut [f64],
        handling: crate::traits::BoundaryHandling,
        rng: &mut dyn RngCore,
    ) {
        super::apply_boundary(position, velocity, |i| self.bounds[i], handling, rng);
    }

    fn decode(&self, raw: &[f64]) -> Vec<f64> {
        raw.to_vec() // identity: the continuous space evaluates in f64 directly
    }

    fn span(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }
}
