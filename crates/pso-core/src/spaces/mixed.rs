//! Mixed search space: each dimension can be real, integer or binary.
//!
//! Like the other spaces, the swarm moves in a continuous interior; the
//! per-dimension type is applied only at `decode`. The scalar type is `f64`:
//! integer and binary dimensions come out as whole-valued floats (e.g. `3.0`,
//! `1.0`), so a single objective signature serves every dimension type.

use rand::{Rng, RngCore};

use crate::spaces::Discretization;
use crate::traits::{BoundaryHandling, SearchSpace};

/// Type of a single decision variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Continuous real variable.
    Real,
    /// Integer variable (rounded at evaluation time).
    Integer,
    /// Binary variable in `{0, 1}`.
    Binary,
}

/// Search box with a per-dimension variable type.
#[derive(Debug, Clone)]
pub struct MixedSpace {
    bounds: Vec<(f64, f64)>,
    types: Vec<VarType>,
    discretization: Discretization,
}

impl MixedSpace {
    /// Creates the space from per-dimension `bounds` and `types` (same length).
    /// Binary dimensions are forced to the `[0, 1]` range.
    ///
    /// # Panics
    /// If `bounds` and `types` differ in length, or a bound has `min > max`.
    pub fn new(mut bounds: Vec<(f64, f64)>, types: Vec<VarType>) -> Self {
        assert_eq!(
            bounds.len(),
            types.len(),
            "bounds and types must have the same length"
        );
        for (i, (lo, hi)) in bounds.iter_mut().enumerate() {
            if types[i] == VarType::Binary {
                *lo = 0.0;
                *hi = 1.0;
            }
            assert!(lo <= hi, "invalid bound in dimension {i}: {lo} > {hi}");
        }
        Self {
            bounds,
            types,
            discretization: Discretization::Round,
        }
    }

    /// Sets the discretization strategy for integer dimensions (chainable).
    pub fn with_discretization(mut self, d: Discretization) -> Self {
        self.discretization = d;
        self
    }

    /// The per-dimension variable types.
    pub fn types(&self) -> &[VarType] {
        &self.types
    }
}

impl SearchSpace for MixedSpace {
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
        handling: BoundaryHandling,
        rng: &mut dyn RngCore,
    ) {
        super::apply_boundary(position, velocity, |i| self.bounds[i], handling, rng);
    }

    fn decode(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter()
            .zip(&self.types)
            .zip(&self.bounds)
            .map(|((&x, &t), &(lo, hi))| match t {
                VarType::Real => x,
                VarType::Integer => {
                    let (lo_i, hi_i) = (lo.round() as i64, hi.round() as i64);
                    self.discretization.apply(x).clamp(lo_i, hi_i) as f64
                }
                VarType::Binary => x.round().clamp(0.0, 1.0),
            })
            .collect()
    }

    fn span(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }
}
