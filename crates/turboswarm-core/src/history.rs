//! Record of the optimization process, for visualization and analysis.
//!
//! It stores the complete trace of the swarm so runs can be animated and
//! convergence curves compared across variants.

/// Complete history of a run.
#[derive(Debug, Clone, Default)]
pub struct History {
    /// Positions per iteration: `positions[iter][particle][dimension]`.
    /// Empty if `record_history == false`.
    pub positions: Vec<Vec<Vec<f64>>>,
    /// Convergence curve: global best value after each iteration.
    pub best_value: Vec<f64>,
    /// Global best position after each iteration.
    pub best_position: Vec<Vec<f64>>,
}

impl History {
    /// Creates an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of iterations recorded in the convergence curve.
    pub fn iterations(&self) -> usize {
        self.best_value.len()
    }
}
