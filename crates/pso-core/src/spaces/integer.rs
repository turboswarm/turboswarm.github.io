//! Integer search space.
//!
//! Approach: the swarm moves in an internal CONTINUOUS space
//! and discretization only happens at `decode`, at evaluation time.
//! This keeps the central problem of integer PSO (how to discretize) explicit
//! and makes it easy to compare strategies.

use rand::{Rng, RngCore};

use crate::traits::SearchSpace;

/// Discretization strategy from continuous -> integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretization {
    /// Rounding to the nearest integer (the most common).
    Round,
    /// Truncation toward zero.
    Truncate,
    /// Floor.
    Floor,
}

impl Discretization {
    pub(crate) fn apply(self, x: f64) -> i64 {
        match self {
            Discretization::Round => x.round() as i64,
            Discretization::Truncate => x.trunc() as i64,
            Discretization::Floor => x.floor() as i64,
        }
    }
}

/// Integer search box defined by `[min, max]` bounds per dimension.
#[derive(Debug, Clone)]
pub struct IntegerSpace {
    bounds: Vec<(i64, i64)>,
    discretization: Discretization,
}

impl IntegerSpace {
    /// Creates the space from the `(min, max)` bounds per dimension.
    /// Panics if any bound has `min > max`.
    pub fn new(bounds: Vec<(i64, i64)>) -> Self {
        for (i, (lo, hi)) in bounds.iter().enumerate() {
            assert!(lo <= hi, "invalid bound in dimension {i}: {lo} > {hi}");
        }
        Self {
            bounds,
            discretization: Discretization::Round,
        }
    }

    /// Changes the discretization strategy (chainable).
    pub fn with_discretization(mut self, d: Discretization) -> Self {
        self.discretization = d;
        self
    }

    /// The integer `(min, max)` bounds per dimension.
    pub fn bounds(&self) -> &[(i64, i64)] {
        &self.bounds
    }
}

impl SearchSpace for IntegerSpace {
    type Scalar = i64;

    fn dim(&self) -> usize {
        self.bounds.len()
    }

    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo as f64..=hi as f64))
            .collect()
    }

    fn sample_velocity(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| {
                let range = (hi - lo) as f64;
                rng.gen_range(-range..=range)
            })
            .collect()
    }

    fn clamp(&self, position: &mut [f64]) {
        for (x, &(lo, hi)) in position.iter_mut().zip(&self.bounds) {
            *x = x.clamp(lo as f64, hi as f64);
        }
    }

    fn enforce_bounds(
        &self,
        position: &mut [f64],
        velocity: &mut [f64],
        handling: crate::traits::BoundaryHandling,
        rng: &mut dyn RngCore,
    ) {
        // The swarm moves in the continuous interior; bounds are the integer
        // limits expressed as floats.
        super::apply_boundary(
            position,
            velocity,
            |i| {
                let (lo, hi) = self.bounds[i];
                (lo as f64, hi as f64)
            },
            handling,
            rng,
        );
    }

    fn decode(&self, raw: &[f64]) -> Vec<i64> {
        raw.iter()
            .zip(&self.bounds)
            .map(|(&x, &(lo, hi))| self.discretization.apply(x).clamp(lo, hi))
            .collect()
    }

    fn span(&self) -> Vec<(f64, f64)> {
        self.bounds.iter().map(|&(lo, hi)| (lo as f64, hi as f64)).collect()
    }
}
