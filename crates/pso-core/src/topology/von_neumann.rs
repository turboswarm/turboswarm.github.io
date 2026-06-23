//! Von Neumann topology: particles are arranged in a 2D `rows × cols` grid
//! and each one is informed by its 4 neighbors (up, down, left, right) with
//! toroidal edges (they wrap around). It is a middle ground between the ring
//! (very local) and `GlobalBest` (fully global) and tends to give good results
//! on multimodal problems.

use crate::swarm::Swarm;
use crate::traits::Topology;

/// `rows × cols` grid. The particle with index `i` occupies cell
/// `(i / cols, i % cols)`. If the swarm has more particles than cells, the
/// extras simply have no position in the grid and are informed only by
/// themselves; it is best to have `rows · cols == n_particles`.
#[derive(Debug, Clone)]
pub struct VonNeumann {
    rows: usize,
    cols: usize,
}

impl VonNeumann {
    /// Creates the `rows × cols` grid. Panics if `rows` or `cols` are 0.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows >= 1 && cols >= 1, "the grid needs rows, cols >= 1");
        Self { rows, cols }
    }

    /// Builds the most square grid possible for `n` particles.
    /// Useful when only the swarm size is known (e.g. from Python).
    pub fn square_for(n: usize) -> Self {
        let mut rows = (n as f64).sqrt() as usize;
        if rows == 0 {
            rows = 1;
        }
        // The number of columns covers the whole swarm (ceiling of the division).
        let cols = n.div_ceil(rows);
        Self::new(rows, cols)
    }
}

impl Topology for VonNeumann {
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize> {
        let n = swarm.len();
        let (r, c) = (i / self.cols, i % self.cols);

        // Toroidal neighbors in the grid.
        let up = ((r + self.rows - 1) % self.rows) * self.cols + c;
        let down = ((r + 1) % self.rows) * self.cols + c;
        let left = r * self.cols + (c + self.cols - 1) % self.cols;
        let right = r * self.cols + (c + 1) % self.cols;

        // The particle itself first (deterministic tie-break). We ignore cells
        // with no particle (if rows·cols > n) and avoid duplicates in degenerate
        // grids (e.g. a single row).
        let mut idx = Vec::with_capacity(5);
        idx.push(i);
        for j in [up, down, left, right] {
            if j < n && !idx.contains(&j) {
                idx.push(j);
            }
        }
        idx
    }
}
