//! Python (PyO3) bindings for the Rust PSO core.
//!
//! Exposes `minimize(...)` and a `PsoResult` class. The function to optimize
//! can be:
//!   - a Python callable (flexible; reacquires the GIL on each evaluation), or
//!   - the NAME of a native Rust benchmark (fast, runs without the GIL).

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use turboswarm_core::benchmarks::{
    ackley, bent_cigar, discus, elliptic, expanded_schaffer, griewank, levy, rastrigin, rosenbrock,
    schwefel, sphere, zakharov,
};
use turboswarm_core::prelude::*;

/// Conversion to f64 for returning positions to Python. Unlike
/// `Into<f64>`, this is implemented for `i64` (we accept the lossy
/// conversion, irrelevant within the range of an optimization problem).
trait ToF64: Copy {
    fn to_f64(self) -> f64;
}
impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl ToF64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Result of an optimization.
///
/// Attributes:
///     best_position (list[float]): best position found, already decoded
///         to the space's type (integers if ``integer=True``).
///     best_value (float): objective function value at ``best_position``.
///     convergence (list[float]): global best value after each iteration (the
///         convergence curve, useful for ``viz.plot_convergence``).
///     history (list[list[list[float]]]): positions of all particles
///         per iteration, indexed ``history[iteration][particle][dimension]``.
///         Empty if run with ``record_history=False``. Required for
///         ``viz.animate_swarm``.
///     evaluations (int): total number of objective evaluations performed.
///     stop_reason (str): why the run stopped — one of ``"max_iterations"``,
///         ``"target"``, ``"max_evaluations"``, ``"stagnation"``, ``"max_time"``.
#[pyclass(name = "PsoResult")]
#[derive(Clone)]
struct PyPsoResult {
    /// Best position found (list of floats).
    #[pyo3(get)]
    best_position: Vec<f64>,
    /// Best objective function value.
    #[pyo3(get)]
    best_value: f64,
    /// Convergence curve: best value per iteration.
    #[pyo3(get)]
    convergence: Vec<f64>,
    /// Positions per iteration: history[iter][particle][dim]. Empty if
    /// record_history=False.
    #[pyo3(get)]
    history: Vec<Vec<Vec<f64>>>,
    /// Total number of objective evaluations performed.
    #[pyo3(get)]
    evaluations: usize,
    /// Why the run stopped (see the class docstring for the values).
    #[pyo3(get)]
    stop_reason: String,
}

#[pymethods]
impl PyPsoResult {
    fn __repr__(&self) -> String {
        format!(
            "PsoResult(best_value={:.6}, best_position={:?}, iters={}, stop_reason={})",
            self.best_value,
            self.best_position,
            self.convergence.len(),
            self.stop_reason
        )
    }
}

/// Resolves a native benchmark by name.
fn native_benchmark(name: &str) -> PyResult<fn(&[f64]) -> f64> {
    match name {
        "sphere" => Ok(sphere as fn(&[f64]) -> f64),
        "rastrigin" => Ok(rastrigin as fn(&[f64]) -> f64),
        "rosenbrock" => Ok(rosenbrock as fn(&[f64]) -> f64),
        "ackley" => Ok(ackley as fn(&[f64]) -> f64),
        "griewank" => Ok(griewank as fn(&[f64]) -> f64),
        "schwefel" => Ok(schwefel as fn(&[f64]) -> f64),
        "bent_cigar" => Ok(bent_cigar as fn(&[f64]) -> f64),
        "discus" => Ok(discus as fn(&[f64]) -> f64),
        "elliptic" => Ok(elliptic as fn(&[f64]) -> f64),
        "zakharov" => Ok(zakharov as fn(&[f64]) -> f64),
        "levy" => Ok(levy as fn(&[f64]) -> f64),
        "expanded_schaffer" => Ok(expanded_schaffer as fn(&[f64]) -> f64),
        other => Err(PyKeyError::new_err(format!(
            "unknown native benchmark: '{other}'. Available: \
             sphere, rastrigin, rosenbrock, ackley, griewank, schwefel, \
             bent_cigar, discus, elliptic, zakharov, levy, expanded_schaffer"
        ))),
    }
}

/// Resolves a native grey benchmark by name (operates on grey numbers).
fn native_grey_benchmark(name: &str) -> PyResult<fn(&[Grey]) -> f64> {
    match name {
        "grey_sphere" => Ok(turboswarm_core::benchmarks::grey_sphere as fn(&[Grey]) -> f64),
        other => Err(PyKeyError::new_err(format!(
            "unknown native grey benchmark: '{other}'. Available: grey_sphere"
        ))),
    }
}

/// Builds the velocity rule by name.
fn build_velocity(name: &str, w: f64, c1: f64, c2: f64) -> PyResult<Box<dyn Velocity>> {
    match name {
        "inertia" => Ok(Box::new(InertiaVelocity::new(w, c1, c2))),
        // Constriction derives χ from c1+c2; it needs c1+c2 > 4. If the user
        // left the default c values (1.49445, sum 2.99), we use the classic
        // Clerc-Kennedy values (2.05) so the formula is well defined.
        "constriction" => {
            let (c1, c2) = if c1 + c2 > 4.0 {
                (c1, c2)
            } else {
                (2.05, 2.05)
            };
            Ok(Box::new(ConstrictionVelocity::new(c1, c2)))
        }
        // FIPS distributes φ = c1 + c2 across all neighbors; it needs φ > 4.
        // With the default c values (sum 2.99) we use the classic φ of 4.1.
        "fips" => {
            let phi = if c1 + c2 > 4.0 { c1 + c2 } else { 4.1 };
            Ok(Box::new(FipsVelocity::new(phi)))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown variant: '{other}'. Available: inertia, constriction, fips"
        ))),
    }
}

/// Parses per-dimension variable type names into `VarType`s.
fn build_var_types(names: &[String]) -> PyResult<Vec<VarType>> {
    names
        .iter()
        .map(|n| match n.as_str() {
            "real" | "float" | "continuous" => Ok(VarType::Real),
            "integer" | "int" => Ok(VarType::Integer),
            "binary" | "bin" => Ok(VarType::Binary),
            other => Err(PyValueError::new_err(format!(
                "unknown variable type: '{other}'. Available: real, integer, binary"
            ))),
        })
        .collect()
}

/// Resolves the `bounds` argument: either a list of `(min, max)` pairs (one per
/// dimension, so each dimension can have its own range) or a single
/// `(min, max)` pair broadcast to `dim` dimensions.
fn resolve_bounds(
    py: Python<'_>,
    bounds: &PyObject,
    dim: Option<usize>,
) -> PyResult<Vec<(f64, f64)>> {
    // List of per-dimension pairs.
    if let Ok(list) = bounds.extract::<Vec<(f64, f64)>>(py) {
        if list.is_empty() {
            return Err(PyValueError::new_err("bounds is empty"));
        }
        if let Some(d) = dim {
            if d != list.len() {
                return Err(PyValueError::new_err(format!(
                    "dim={d} does not match the {} bound pairs given",
                    list.len()
                )));
            }
        }
        return Ok(list);
    }
    // 2D array-like: a NumPy array of shape (dim, 2) or a list of [min, max].
    if let Ok(rows) = bounds.extract::<Vec<Vec<f64>>>(py) {
        if rows.is_empty() {
            return Err(PyValueError::new_err("bounds is empty"));
        }
        let mut pairs = Vec::with_capacity(rows.len());
        for row in &rows {
            if row.len() != 2 {
                return Err(PyValueError::new_err(
                    "each bound must have exactly two values, (min, max)",
                ));
            }
            pairs.push((row[0], row[1]));
        }
        if let Some(d) = dim {
            if d != pairs.len() {
                return Err(PyValueError::new_err(format!(
                    "dim={d} does not match the {} bound pairs given",
                    pairs.len()
                )));
            }
        }
        return Ok(pairs);
    }
    // Single pair broadcast to `dim` dimensions.
    if let Ok(pair) = bounds.extract::<(f64, f64)>(py) {
        match dim {
            Some(d) if d >= 1 => return Ok(vec![pair; d]),
            Some(_) => return Err(PyValueError::new_err("dim must be >= 1")),
            None => {
                return Err(PyValueError::new_err(
                    "with a single (min, max) bound, pass dim=<number of dimensions>",
                ))
            }
        }
    }
    // Single pair as a 2-element array-like (e.g. a NumPy array of shape (2,)).
    if let Ok(flat) = bounds.extract::<Vec<f64>>(py) {
        if flat.len() == 2 {
            let pair = (flat[0], flat[1]);
            match dim {
                Some(d) if d >= 1 => return Ok(vec![pair; d]),
                Some(_) => return Err(PyValueError::new_err("dim must be >= 1")),
                None => {
                    return Err(PyValueError::new_err(
                        "with a single (min, max) bound, pass dim=<number of dimensions>",
                    ))
                }
            }
        }
    }
    Err(PyValueError::new_err(
        "bounds must be a list of (min, max) pairs (or a NumPy array of shape \
         (dim, 2)), or a single (min, max) pair with dim=<n>",
    ))
}

/// Maps a boundary-handling name to its strategy.
fn build_boundary(name: &str) -> PyResult<BoundaryHandling> {
    match name {
        "clamp" => Ok(BoundaryHandling::Clamp),
        "reflect" => Ok(BoundaryHandling::Reflect),
        "wrap" => Ok(BoundaryHandling::Wrap),
        "reinit" => Ok(BoundaryHandling::Reinit),
        other => Err(PyValueError::new_err(format!(
            "unknown bounds_handling: '{other}'. Available: clamp, reflect, wrap, reinit"
        ))),
    }
}

/// Builds the topology by name. `n_particles` allows sizing the Von Neumann
/// grid automatically; `seed` seeds the random topology's internal RNG.
fn build_topology(
    name: &str,
    n_particles: usize,
    seed: Option<u64>,
) -> PyResult<Box<dyn Topology>> {
    match name {
        "global" => Ok(Box::new(GlobalBest::new())),
        "ring" => Ok(Box::new(Ring::lbest())),
        "vonneumann" | "von_neumann" => Ok(Box::new(VonNeumann::square_for(n_particles))),
        "random" => Ok(Box::new(Random::new(3, seed.unwrap_or(0)))),
        other => Err(PyValueError::new_err(format!(
            "unknown topology: '{other}'. Available: global, ring, vonneumann, random"
        ))),
    }
}

/// Minimizes an objective function with Particle Swarm Optimization.
///
/// The function to optimize can be a Python *callable* or the name of a
/// native Rust benchmark (faster, runs without the GIL).
///
/// Args:
///     objective: callable ``f(list[float]) -> float`` to minimize, OR the name
///         (str) of a native benchmark: ``"sphere"``, ``"rastrigin"``,
///         ``"rosenbrock"``, ``"ackley"``, ``"griewank"``, ``"schwefel"``.
///     bounds: either a list of ``(min, max)`` pairs (one per dimension, so each
///         dimension can have its own range) or a single ``(min, max)`` pair
///         used for every dimension (then pass ``dim``).
///     integer (bool): if ``True``, optimizes over integer variables (the
///         position is discretized by rounding when evaluated). Defaults to ``False``.
///     n_particles (int): swarm size. Defaults to 30.
///     max_iter (int): number of iterations. Defaults to 100.
///     w (float): inertia weight (``"inertia"`` variant). Defaults to 0.729.
///     c1 (float): cognitive coefficient (attraction to the personal best).
///     c2 (float): social coefficient (attraction to the neighborhood best).
///     velocity (str): velocity variant: ``"inertia"``, ``"constriction"``
///         or ``"fips"``. Defaults to ``"inertia"``. ``"constriction"`` and
///         ``"fips"`` derive their factor from ``c1 + c2`` (use 2.05/4.1 if the
///         sum does not exceed 4).
///     topology (str): social structure: ``"global"``, ``"ring"``,
///         ``"vonneumann"`` or ``"random"``. Defaults to ``"global"``. FIPS
///         performs better with local topologies (``"ring"``, ``"vonneumann"``).
///     seed (int | None): RNG seed. Fix it for reproducible runs. Defaults to
///         ``None`` (system seed).
///     record_history (bool): if ``True`` (the default), stores the swarm
///         trace for visualization. Disable it to save memory and time.
///     v_max (float | None): if set, clamps every velocity component to
///         ``[-v_max, v_max]`` after each update. Defaults to ``None`` (off).
///     patience (int): early stopping. Stop when the global best does not
///         improve by more than ``tol`` for ``patience`` consecutive
///         iterations. ``0`` (default) disables it (always runs ``max_iter``).
///     tol (float): minimum improvement counted as progress for the
///         ``patience`` window. Defaults to ``0.0``.
///     constraints (list[callable] | None): inequality constraints, each a
///         callable ``g(x) -> float`` that is feasible when ``g(x) <= 0``. A
///         quadratic penalty ``penalty * sum(max(0, g(x))**2)`` is added to the
///         objective. Requires ``objective`` to be a Python callable (not a
///         native benchmark name). Defaults to ``None``.
///     penalty (float): weight of the constraint penalty (shared by the
///         inequality and equality terms). Defaults to ``1e6``.
///     equality_constraints (list[callable] | None): equality constraints, each
///         a callable ``h(x) -> float`` that is feasible when ``h(x) == 0``. A
///         quadratic penalty ``penalty * sum(h(x)**2)`` is added to the
///         objective. Like ``constraints``, requires a Python objective (not a
///         native benchmark). Defaults to ``None``. Note: equality constraints
///         are sensitive to ``penalty`` — a very large value (e.g. the ``1e6``
///         default) makes the feasible valley so steep the swarm converges onto
///         a feasible but sub-optimal point; moderate weights (``1e3``-``1e4``)
///         usually balance feasibility and objective better. For hard equalities
///         a ``repair`` operator is often more robust.
///     repair (callable | None): a repair operator ``repair(x) -> x'`` applied
///         to each candidate before it is evaluated, mapping infeasible points
///         back to (or towards) the feasible region. The objective and the
///         constraints see the repaired point, and the returned
///         ``best_position`` is repaired too, so the reported solution is
///         consistent with its value. Requires a Python objective. Defaults to
///         ``None``.
///     binary (bool): if ``True``, optimize over binary variables ``{0, 1}``
///         (a {0,1} integer space); the dimension is taken from ``bounds``.
///         Defaults to ``False``.
///     max_evals (int | None): stop after this many objective evaluations.
///     target (float | None): stop as soon as the best value is ``<= target``.
///     max_time (float | None): stop after this many seconds of wall-clock time.
///     callback (callable | None): called once per iteration as
///         ``callback(iteration, best_value)``. Return ``False`` to stop early
///         (reported as ``stop_reason="callback"``); ``None`` or any other
///         return continues. Useful for live logging or custom stopping.
///     bounds_handling (str): what to do with particles that leave the bounds:
///         ``"clamp"`` (default), ``"reflect"``, ``"wrap"`` or ``"reinit"``.
///     vectorized (bool): if ``True``, ``objective`` receives the WHOLE swarm
///         per call as a NumPy array (shape ``n_particles x dim``) and must
///         return one value per row. This lets a NumPy-vectorized objective
///         amortize its overhead over the swarm. Uses synchronous updates;
///         incompatible with native benchmarks, constraints and callback.
///         Defaults to ``False``.
///     var_types (list[str] | None): per-dimension variable types for a mixed
///         problem — each ``"real"``, ``"integer"`` or ``"binary"`` (same
///         length as ``bounds``). Integer/binary dimensions come back as
///         whole-valued floats. Takes precedence over ``integer``/``binary``.
///     dim (int | None): number of dimensions when ``bounds`` is a single
///         ``(min, max)`` pair (ignored when ``bounds`` is already a list).
///
/// Returns:
///     PsoResult: with ``best_position``, ``best_value``, ``convergence`` and
///     ``history``.
///
/// Raises:
///     KeyError: if the name of a non-existent native benchmark is passed.
///     ValueError: if the variant or topology does not exist.
///     Exception: propagates any error raised by the Python ``objective``.
///
/// Example:
///     >>> import turboswarm as pso
///     >>> r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2, seed=42)
///     >>> r.best_value < 1e-3
///     True
///     >>> r = pso.minimize(lambda x: sum(xi*xi for xi in x),
///     ...                  bounds=[(-5, 5)] * 3, velocity="fips", topology="ring")
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    objective,
    bounds,
    integer = false,
    n_particles = 30,
    max_iter = 100,
    w = 0.729,
    c1 = 1.49445,
    c2 = 1.49445,
    velocity = "inertia",
    topology = "global",
    seed = None,
    record_history = true,
    v_max = None,
    patience = 0,
    tol = 0.0,
    constraints = None,
    penalty = 1e6,
    binary = false,
    max_evals = None,
    target = None,
    max_time = None,
    callback = None,
    bounds_handling = "clamp",
    vectorized = false,
    var_types = None,
    dim = None,
    equality_constraints = None,
    repair = None,
))]
fn minimize(
    py: Python<'_>,
    objective: PyObject,
    bounds: PyObject,
    integer: bool,
    n_particles: usize,
    max_iter: usize,
    w: f64,
    c1: f64,
    c2: f64,
    velocity: &str,
    topology: &str,
    seed: Option<u64>,
    record_history: bool,
    v_max: Option<f64>,
    patience: usize,
    tol: f64,
    constraints: Option<Vec<PyObject>>,
    penalty: f64,
    binary: bool,
    max_evals: Option<usize>,
    target: Option<f64>,
    max_time: Option<f64>,
    callback: Option<PyObject>,
    bounds_handling: &str,
    vectorized: bool,
    var_types: Option<Vec<String>>,
    dim: Option<usize>,
    equality_constraints: Option<Vec<PyObject>>,
    repair: Option<PyObject>,
) -> PyResult<PyPsoResult> {
    let bounds = resolve_bounds(py, &bounds, dim)?;
    let params = PsoParams {
        w,
        c1,
        c2,
        n_particles,
        max_iterations: max_iter,
        seed,
        record_history,
        v_max,
        patience,
        tol,
        max_evals,
        target,
        max_time: max_time.map(std::time::Duration::from_secs_f64),
        bounds_handling: build_boundary(bounds_handling)?,
    };
    let vel = build_velocity(velocity, w, c1, c2)?;
    let topo = build_topology(topology, n_particles, seed)?;

    // Is the objective function a string (native benchmark) or a callable?
    let native_name: Option<String> = objective.extract(py).ok();

    let constraints = constraints.unwrap_or_default();
    let equality = equality_constraints.unwrap_or_default();
    let needs_python_obj = !constraints.is_empty() || !equality.is_empty() || repair.is_some();
    if needs_python_obj && native_name.is_some() {
        return Err(PyValueError::new_err(
            "constraints, equality_constraints and repair require a Python \
             objective callable, not a native benchmark name; pass a function \
             as the objective",
        ));
    }

    if let Some(names) = var_types {
        // Per-dimension types: real / integer / binary.
        if names.len() != bounds.len() {
            return Err(PyValueError::new_err(format!(
                "var_types has length {} but bounds has length {}",
                names.len(),
                bounds.len()
            )));
        }
        let types = build_var_types(&names)?;
        let space = MixedSpace::new(bounds, types);
        // Mixed positions decode to floats (integer dims are whole-valued).
        run(
            py,
            space,
            vel,
            topo,
            params,
            objective,
            native_name,
            false,
            constraints,
            penalty,
            equality,
            repair,
            callback,
            vectorized,
        )
    } else if integer || binary {
        // `binary` is the {0, 1} special case of an integer space.
        let int_bounds: Vec<(i64, i64)> = if binary {
            vec![(0, 1); bounds.len()]
        } else {
            bounds
                .iter()
                .map(|&(lo, hi)| (lo.round() as i64, hi.round() as i64))
                .collect()
        };
        let space = IntegerSpace::new(int_bounds);
        run(
            py,
            space,
            vel,
            topo,
            params,
            objective,
            native_name,
            true,
            constraints,
            penalty,
            equality,
            repair,
            callback,
            vectorized,
        )
    } else {
        let space = ContinuousSpace::new(bounds);
        run(
            py,
            space,
            vel,
            topo,
            params,
            objective,
            native_name,
            false,
            constraints,
            penalty,
            equality,
            repair,
            callback,
            vectorized,
        )
    }
}

/// Logic common to both spaces. `S::Scalar` is converted to `f64` to
/// always return the position as a list of floats to Python.
#[allow(clippy::too_many_arguments)]
fn run<S>(
    py: Python<'_>,
    space: S,
    vel: Box<dyn Velocity>,
    topo: Box<dyn Topology>,
    params: PsoParams,
    objective: PyObject,
    native_name: Option<String>,
    is_integer: bool,
    constraints: Vec<PyObject>,
    penalty: f64,
    equality: Vec<PyObject>,
    repair: Option<PyObject>,
    callback: Option<PyObject>,
    vectorized: bool,
) -> PyResult<PyPsoResult>
where
    S: SearchSpace,
    S::Scalar: ToF64,
{
    use std::cell::RefCell;

    let pso = Pso::new(space, vel, topo, params);
    // Resolve the native benchmark once (constraints are rejected for it).
    let native_fn = match &native_name {
        Some(name) => Some(native_benchmark(name)?),
        None => None,
    };
    // Shared so both the objective and the callback closures can record a
    // Python error and abort the run.
    let call_error: RefCell<Option<PyErr>> = RefCell::new(None);

    // Vectorized path: one call per iteration with the whole swarm.
    if vectorized {
        if native_fn.is_some() {
            return Err(PyValueError::new_err(
                "vectorized=True requires a Python objective callable, not a native benchmark",
            ));
        }
        if !constraints.is_empty() || !equality.is_empty() || repair.is_some() || callback.is_some()
        {
            return Err(PyValueError::new_err(
                "vectorized=True does not support constraints, equality_constraints, \
                 repair or callback yet",
            ));
        }
        // The swarm is passed as a contiguous NumPy array (one buffer copy, no
        // per-element Python objects), and the result is read back as a slice.
        let batch = |positions: &[Vec<S::Scalar>]| -> Vec<f64> {
            let n = positions.len();
            if call_error.borrow().is_some() {
                return vec![f64::INFINITY; n];
            }
            let dim = positions.first().map_or(0, |p| p.len());
            let flat: Vec<f64> = positions
                .iter()
                .flat_map(|p| p.iter().map(|&v| v.to_f64()))
                .collect();
            let arr = numpy::ndarray::Array2::from_shape_vec((n, dim), flat)
                .expect("n*dim matches the flat buffer");
            let pyarr = arr.into_pyarray_bound(py);

            let out = match objective.call1(py, (pyarr,)) {
                Ok(o) => o,
                Err(e) => {
                    *call_error.borrow_mut() = Some(e);
                    return vec![f64::INFINITY; n];
                }
            };
            // Fast path: a NumPy array result is read as a contiguous slice.
            // Fall back to a generic sequence (e.g. a Python list).
            let vals: Vec<f64> = if let Ok(a) = out.extract::<PyReadonlyArray1<f64>>(py) {
                a.as_array().iter().copied().collect()
            } else {
                match out.extract::<Vec<f64>>(py) {
                    Ok(v) => v,
                    Err(e) => {
                        *call_error.borrow_mut() = Some(e);
                        return vec![f64::INFINITY; n];
                    }
                }
            };
            if vals.len() != n {
                *call_error.borrow_mut() = Some(PyValueError::new_err(
                    "vectorized objective must return one value per row",
                ));
                return vec![f64::INFINITY; n];
            }
            vals
        };
        let result = pso.minimize_batch(batch);
        if let Some(e) = call_error.into_inner() {
            return Err(e);
        }
        return Ok(to_py_result(result));
    }

    // Single objective closure: native path stays GIL-free; the Python path
    // reacquires the GIL and applies the constraint penalty.
    let eval = |x: &[S::Scalar]| -> f64 {
        if call_error.borrow().is_some() {
            return f64::INFINITY;
        }
        if let Some(f) = native_fn {
            let xf: Vec<f64> = x.iter().map(|&v| v.to_f64()).collect();
            return f(&xf);
        }
        // Build the argument list. With a repair operator, the candidate is
        // mapped to the feasible region first and everything downstream (the
        // objective and the constraints) sees the repaired point.
        let mk_args = |vals: &[f64]| -> Bound<'_, PyAny> {
            if is_integer {
                let xs: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
                PyList::new_bound(py, xs).into_any()
            } else {
                PyList::new_bound(py, vals.to_vec()).into_any()
            }
        };
        let raw_vals: Vec<f64> = x.iter().map(|&v| v.to_f64()).collect();
        let args = match &repair {
            None => mk_args(&raw_vals),
            Some(r) => match r
                .call1(py, (mk_args(&raw_vals),))
                .and_then(|o| o.extract::<Vec<f64>>(py))
            {
                Ok(repaired) => mk_args(&repaired),
                Err(e) => {
                    *call_error.borrow_mut() = Some(e);
                    return f64::INFINITY;
                }
            },
        };
        let base = match objective
            .call1(py, (args.clone(),))
            .and_then(|r| r.extract::<f64>(py))
        {
            Ok(val) => val,
            Err(e) => {
                *call_error.borrow_mut() = Some(e);
                return f64::INFINITY;
            }
        };
        let mut penalty_term = 0.0;
        // Inequality g(x) <= 0: penalize the positive violation, squared.
        for g in &constraints {
            match g
                .call1(py, (args.clone(),))
                .and_then(|r| r.extract::<f64>(py))
            {
                Ok(gv) => {
                    let viol = gv.max(0.0);
                    penalty_term += viol * viol;
                }
                Err(e) => {
                    *call_error.borrow_mut() = Some(e);
                    return f64::INFINITY;
                }
            }
        }
        // Equality h(x) == 0: penalize the squared deviation from zero.
        for h in &equality {
            match h
                .call1(py, (args.clone(),))
                .and_then(|r| r.extract::<f64>(py))
            {
                Ok(hv) => {
                    penalty_term += hv * hv;
                }
                Err(e) => {
                    *call_error.borrow_mut() = Some(e);
                    return f64::INFINITY;
                }
            }
        }
        base + penalty * penalty_term
    };

    let result = if let Some(cb) = callback {
        // Called once per iteration with (iteration, best_value). Returning
        // False stops the run; None / non-bool returns continue.
        let cb_closure = |info: &IterationInfo| -> bool {
            if call_error.borrow().is_some() {
                return false;
            }
            match cb.call1(py, (info.iteration, info.best_value)) {
                Ok(ret) => ret.extract::<bool>(py).unwrap_or(true),
                Err(e) => {
                    *call_error.borrow_mut() = Some(e);
                    false
                }
            }
        };
        pso.minimize_with_callback(eval, cb_closure)
    } else {
        pso.minimize(eval)
    };

    if let Some(e) = call_error.into_inner() {
        return Err(e);
    }
    let mut out = to_py_result(result);
    // Report the repaired solution so best_position is consistent with the
    // best_value the search actually optimized (which already used the repair).
    if let Some(r) = &repair {
        let raw = if is_integer {
            let xs: Vec<i64> = out.best_position.iter().map(|&v| v as i64).collect();
            PyList::new_bound(py, xs).into_any()
        } else {
            PyList::new_bound(py, out.best_position.clone()).into_any()
        };
        out.best_position = r.call1(py, (raw,))?.extract::<Vec<f64>>(py)?;
    }
    Ok(out)
}

fn to_py_result<T>(r: PsoResult<T>) -> PyPsoResult
where
    T: ToF64,
{
    PyPsoResult {
        best_position: r.best_position.iter().map(|&v| v.to_f64()).collect(),
        best_value: r.best_value,
        convergence: r.history.best_value.clone(),
        history: r.history.positions,
        evaluations: r.evaluations,
        stop_reason: r.stop_reason.as_str().to_string(),
    }
}

/// Result of a grey optimization returned by `minimize_grey`.
///
/// Attributes:
///     best_position (list[tuple[float, float]]): the best grey vector, one
///         ``(lower, upper)`` interval per grey variable.
///     best_centers (list[float]): center of each grey variable (``(lo+hi)/2``).
///     best_spreads (list[float]): half-width of each grey variable
///         (``(hi-lo)/2``, always ``>= 0``).
///     best_value (float): objective value at ``best_position`` (the
///         whitenized scalar the search minimized).
///     convergence (list[float]): best value after each iteration.
///     evaluations (int): total objective evaluations performed.
///     stop_reason (str): why the run stopped (see ``PsoResult``).
#[pyclass(name = "GreyResult")]
#[derive(Clone)]
struct PyGreyResult {
    /// Best grey vector as `(lower, upper)` intervals.
    #[pyo3(get)]
    best_position: Vec<(f64, f64)>,
    /// Center of each grey variable.
    #[pyo3(get)]
    best_centers: Vec<f64>,
    /// Half-width (spread) of each grey variable.
    #[pyo3(get)]
    best_spreads: Vec<f64>,
    /// Best objective value (whitenized scalar).
    #[pyo3(get)]
    best_value: f64,
    /// Convergence curve: best value per iteration.
    #[pyo3(get)]
    convergence: Vec<f64>,
    /// Total number of objective evaluations performed.
    #[pyo3(get)]
    evaluations: usize,
    /// Why the run stopped.
    #[pyo3(get)]
    stop_reason: String,
}

#[pymethods]
impl PyGreyResult {
    fn __repr__(&self) -> String {
        format!(
            "GreyResult(best_value={:.6}, best_position={:?}, iters={}, stop_reason={})",
            self.best_value,
            self.best_position,
            self.convergence.len(),
            self.stop_reason
        )
    }
}

/// Resolves the `max_spread` argument: ``None`` (no extra cap), a single float
/// broadcast to every grey variable, or one value per variable (matching `n`).
fn resolve_max_spread(
    py: Python<'_>,
    max_spread: &Option<PyObject>,
    n: usize,
) -> PyResult<Vec<f64>> {
    let Some(max_spread) = max_spread else {
        // No extra cap: the spread is limited only by the (lower, upper) box.
        return Ok(vec![f64::INFINITY; n]);
    };
    if let Ok(per_var) = max_spread.extract::<Vec<f64>>(py) {
        if per_var.len() != n {
            return Err(PyValueError::new_err(format!(
                "max_spread has {} values but there are {n} grey variables",
                per_var.len()
            )));
        }
        return Ok(per_var);
    }
    if let Ok(s) = max_spread.extract::<f64>(py) {
        return Ok(vec![s; n]);
    }
    Err(PyValueError::new_err(
        "max_spread must be a float, a list of floats (one per grey variable), or None",
    ))
}

/// Minimizes a grey objective with PSO. Each variable is a *grey number*
/// ⊗ = ``[lower, upper]`` constrained to lie within its ``bounds``; the swarm
/// searches over its center and spread.
///
/// The objective can be a Python *callable* that receives the candidate as a
/// ``list[tuple[float, float]]`` (one pair per grey variable, in the
/// ``representation`` you choose) and returns a single ``float`` (the
/// whitenized scalar to minimize — e.g. via interval arithmetic and a
/// whitenization rule of your choice), OR the name (str) of a native grey
/// benchmark (``"grey_sphere"``), which runs without the GIL.
///
/// Args:
///     objective: callable ``f(list[tuple[float, float]]) -> float`` to minimize,
///         OR the name (str) of a native grey benchmark: ``"grey_sphere"``.
///     bounds: ``(lower, upper)`` LIMITS each grey number's interval must stay
///         within — either a list of pairs (one per variable) or a single pair
///         with ``dim``. The whole decoded interval is kept inside these limits.
///     max_spread: optional extra cap on the half-width of each grey variable —
///         ``None`` (default, limited only by ``bounds``), a single float
///         (broadcast) or a list of floats (one per variable).
///     representation (str): how each grey number is passed to (and read from)
///         the objective: ``"interval"`` (default) gives ``(lower, upper)``
///         pairs; ``"center_spread"`` gives ``(center, spread)`` pairs. Does not
///         affect native benchmarks. The result always exposes both forms.
///     n_particles, max_iter, w, c1, c2, velocity, topology, seed,
///     record_history, v_max, patience, tol, max_evals, target, max_time,
///     dim: same meaning as in ``minimize``. (Grey bounds are enforced by
///     projection onto the feasible region, so ``bounds_handling`` does not
///     apply.)
///
/// Returns:
///     GreyResult: with ``best_position`` (intervals), ``best_centers``,
///     ``best_spreads``, ``best_value`` and ``convergence``.
///
/// Example:
///     >>> import turboswarm as pso
///     >>> # Find grey numbers whose midpoints minimize a sphere while staying crisp.
///     >>> def f(greys):
///     ...     centers = [(lo + hi) / 2 for (lo, hi) in greys]
///     ...     spreads = [(hi - lo) / 2 for (lo, hi) in greys]
///     ...     return sum(c * c for c in centers) + sum(spreads)
///     >>> r = pso.minimize_grey(f, bounds=(-5, 5), dim=2, seed=42)
///     >>> r.best_value < 1e-2
///     True
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    objective,
    bounds,
    max_spread = None,
    representation = "interval",
    n_particles = 30,
    max_iter = 100,
    w = 0.729,
    c1 = 1.49445,
    c2 = 1.49445,
    velocity = "inertia",
    topology = "global",
    seed = None,
    record_history = true,
    v_max = None,
    patience = 0,
    tol = 0.0,
    max_evals = None,
    target = None,
    max_time = None,
    dim = None,
))]
fn minimize_grey(
    py: Python<'_>,
    objective: PyObject,
    bounds: PyObject,
    max_spread: Option<PyObject>,
    representation: &str,
    n_particles: usize,
    max_iter: usize,
    w: f64,
    c1: f64,
    c2: f64,
    velocity: &str,
    topology: &str,
    seed: Option<u64>,
    record_history: bool,
    v_max: Option<f64>,
    patience: usize,
    tol: f64,
    max_evals: Option<usize>,
    target: Option<f64>,
    max_time: Option<f64>,
    dim: Option<usize>,
) -> PyResult<PyGreyResult> {
    let center_spread = match representation {
        "interval" => false,
        "center_spread" | "center-spread" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown representation: '{other}'. Available: interval, center_spread"
            )))
        }
    };
    let center_bounds = resolve_bounds(py, &bounds, dim)?;
    let max_spread = resolve_max_spread(py, &max_spread, center_bounds.len())?;
    let params = PsoParams {
        w,
        c1,
        c2,
        n_particles,
        max_iterations: max_iter,
        seed,
        record_history,
        v_max,
        patience,
        tol,
        max_evals,
        target,
        max_time: max_time.map(std::time::Duration::from_secs_f64),
        // Grey bounds are enforced by projection (see GreySpace::clamp), so the
        // boundary-handling strategy is irrelevant here; keep the default.
        bounds_handling: BoundaryHandling::Clamp,
    };
    let vel = build_velocity(velocity, w, c1, c2)?;
    let topo = build_topology(topology, n_particles, seed)?;
    let space = GreySpace::new(center_bounds, max_spread);

    use std::cell::RefCell;
    let pso = Pso::new(space, vel, topo, params);
    let call_error: RefCell<Option<PyErr>> = RefCell::new(None);

    // The objective is either the name of a native grey benchmark (GIL-free) or
    // a Python callable that takes a list of (lower, upper) tuples.
    let native_fn = match objective.extract::<String>(py) {
        Ok(name) => Some(native_grey_benchmark(&name)?),
        Err(_) => None,
    };

    let eval = |g: &[Grey]| -> f64 {
        if let Some(f) = native_fn {
            return f(g);
        }
        if call_error.borrow().is_some() {
            return f64::INFINITY;
        }
        // Present each grey number in the chosen representation.
        let pairs: Vec<(f64, f64)> = if center_spread {
            g.iter().map(|gi| (gi.center(), gi.spread())).collect()
        } else {
            g.iter().map(|gi| (gi.lower(), gi.upper())).collect()
        };
        let args = PyList::new_bound(py, pairs);
        match objective
            .call1(py, (args,))
            .and_then(|r| r.extract::<f64>(py))
        {
            Ok(v) => v,
            Err(e) => {
                *call_error.borrow_mut() = Some(e);
                f64::INFINITY
            }
        }
    };

    let result = pso.minimize(eval);
    if let Some(e) = call_error.into_inner() {
        return Err(e);
    }

    let best_position: Vec<(f64, f64)> = result
        .best_position
        .iter()
        .map(|g| (g.lower(), g.upper()))
        .collect();
    let best_centers = result.best_position.iter().map(|g| g.center()).collect();
    let best_spreads = result.best_position.iter().map(|g| g.spread()).collect();
    Ok(PyGreyResult {
        best_position,
        best_centers,
        best_spreads,
        best_value: result.best_value,
        convergence: result.history.best_value.clone(),
        evaluations: result.evaluations,
        stop_reason: result.stop_reason.as_str().to_string(),
    })
}

/// An approximated Pareto front returned by `minimize_multi`.
///
/// Attributes:
///     positions (list[list[float]]): the non-dominated decision vectors.
///     objectives (list[list[float]]): their objective values (same order).
#[pyclass(name = "ParetoFront")]
#[derive(Clone)]
struct PyParetoFront {
    /// Non-dominated decision vectors.
    #[pyo3(get)]
    positions: Vec<Vec<f64>>,
    /// Objective values of each solution.
    #[pyo3(get)]
    objectives: Vec<Vec<f64>>,
}

#[pymethods]
impl PyParetoFront {
    fn __len__(&self) -> usize {
        self.positions.len()
    }
    fn __repr__(&self) -> String {
        format!("ParetoFront(size={})", self.positions.len())
    }

    /// Hypervolume of this front: the volume of objective space dominated by it
    /// and bounded by ``reference`` (minimization — larger is better). The
    /// standard single-indicator measure of front quality (convergence + spread).
    ///
    /// Args:
    ///     reference (list[float] | None): the bounding "worst" point, with one
    ///         value per objective. Each value must exceed the front in that
    ///         objective. If ``None``, it is derived from the front's nadir
    ///         (worst value per objective + 10% of its spread); convenient for a
    ///         single run but **not** comparable across runs — pass an explicit,
    ///         shared reference when comparing fronts.
    ///
    /// Returns:
    ///     float: the hypervolume (``0.0`` for an empty front).
    #[pyo3(signature = (reference = None))]
    fn hypervolume(&self, reference: Option<Vec<f64>>) -> PyResult<f64> {
        if self.objectives.is_empty() {
            return Ok(0.0);
        }
        let n_obj = self.objectives[0].len();
        let reference = match reference {
            Some(r) => {
                if r.len() != n_obj {
                    return Err(PyValueError::new_err(format!(
                        "reference must have one value per objective ({n_obj}), got {}",
                        r.len()
                    )));
                }
                r
            }
            None => turboswarm_core::mopso::nadir_reference(&self.objectives),
        };
        Ok(turboswarm_core::mopso::hypervolume(
            &self.objectives,
            &reference,
        ))
    }
}

/// Hypervolume of an arbitrary set of objective vectors (minimization).
///
/// A free-standing version of :meth:`ParetoFront.hypervolume` for fronts not
/// produced by ``minimize_multi`` (e.g. a reference front, or one loaded from
/// disk for benchmarking).
///
/// Args:
///     front (list[list[float]]): objective vectors, one per solution.
///     reference (list[float]): the bounding "worst" point, one value per
///         objective; each value must exceed the front in that objective.
///
/// Returns:
///     float: the hypervolume dominated by ``front`` under ``reference``.
#[pyfunction(name = "hypervolume")]
fn py_hypervolume(front: Vec<Vec<f64>>, reference: Vec<f64>) -> f64 {
    turboswarm_core::mopso::hypervolume(&front, &reference)
}

/// Common multi-objective driver. Evaluates a Python callable that returns a
/// list of objective values, and returns the Pareto front.
fn run_multi<S>(
    py: Python<'_>,
    space: S,
    vel: Box<dyn Velocity>,
    params: MopsoParams,
    objective: PyObject,
    is_integer: bool,
) -> PyResult<PyParetoFront>
where
    S: SearchSpace,
    S::Scalar: ToF64,
{
    use std::cell::{Cell, RefCell};

    let mopso = Mopso::new(space, vel, params);
    let call_error: RefCell<Option<PyErr>> = RefCell::new(None);
    let n_obj = Cell::new(1usize); // remembered objective count for error fills

    let obj = |x: &[S::Scalar]| -> Vec<f64> {
        if call_error.borrow().is_some() {
            return vec![f64::INFINITY; n_obj.get()];
        }
        let args = if is_integer {
            let xs: Vec<i64> = x.iter().map(|&v| v.to_f64() as i64).collect();
            PyList::new_bound(py, xs).into_any()
        } else {
            let xs: Vec<f64> = x.iter().map(|&v| v.to_f64()).collect();
            PyList::new_bound(py, xs).into_any()
        };
        match objective
            .call1(py, (args,))
            .and_then(|r| r.extract::<Vec<f64>>(py))
        {
            Ok(v) => {
                n_obj.set(v.len());
                v
            }
            Err(e) => {
                *call_error.borrow_mut() = Some(e);
                vec![f64::INFINITY; n_obj.get()]
            }
        }
    };

    let result = mopso.minimize(obj);
    if let Some(e) = call_error.into_inner() {
        return Err(e);
    }
    let positions = result
        .front
        .iter()
        .map(|s| s.position.iter().map(|&v| v.to_f64()).collect())
        .collect();
    let objectives = result.front.iter().map(|s| s.objectives.clone()).collect();
    Ok(PyParetoFront {
        positions,
        objectives,
    })
}

/// Multi-objective optimization (MOPSO). Returns the Pareto front.
///
/// Args:
///     objective: callable ``f(list) -> list[float]`` returning one value per
///         objective (all minimized).
///     bounds (list[tuple[float, float]]): ``(min, max)`` per dimension.
///     n_particles (int): swarm size. Defaults to 100.
///     max_iter (int): iterations. Defaults to 100.
///     archive_size (int): maximum Pareto-front size. Defaults to 100.
///     w, c1, c2 (float): velocity coefficients.
///     velocity (str): ``"inertia"`` or ``"constriction"`` (single-leader
///         rules; ``"fips"`` is not applicable to MOPSO).
///     seed (int | None): RNG seed.
///     integer (bool), binary (bool), var_types (list[str] | None): same
///         meaning as in ``minimize``.
///     mutation_rate (float): turbulence strength in [0, 1] (default 0.1);
///         improves front spread. ``0`` disables it.
///     grid_divisions (int | None): archive diversity strategy. ``None``
///         (default) keeps the most isolated members by crowding distance;
///         an int ``d`` uses Coello's adaptive hypercube grid with ``d``
///         divisions per objective (pruning drops members from the most crowded
///         cell, leaders are drawn towards sparser cells), which tends to spread
///         the front more evenly.
///
/// Returns:
///     ParetoFront: with ``positions`` and ``objectives``.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    objective,
    bounds,
    n_particles = 100,
    max_iter = 100,
    archive_size = 100,
    w = 0.729,
    c1 = 1.49445,
    c2 = 1.49445,
    velocity = "inertia",
    seed = None,
    integer = false,
    binary = false,
    var_types = None,
    mutation_rate = 0.1,
    dim = None,
    grid_divisions = None,
))]
fn minimize_multi(
    py: Python<'_>,
    objective: PyObject,
    bounds: PyObject,
    n_particles: usize,
    max_iter: usize,
    archive_size: usize,
    w: f64,
    c1: f64,
    c2: f64,
    velocity: &str,
    seed: Option<u64>,
    integer: bool,
    binary: bool,
    var_types: Option<Vec<String>>,
    mutation_rate: f64,
    dim: Option<usize>,
    grid_divisions: Option<usize>,
) -> PyResult<PyParetoFront> {
    if velocity == "fips" {
        return Err(PyValueError::new_err(
            "MOPSO needs a single-leader velocity ('inertia' or 'constriction'); 'fips' does not apply",
        ));
    }
    let bounds = resolve_bounds(py, &bounds, dim)?;
    let vel = build_velocity(velocity, w, c1, c2)?;
    let params = MopsoParams {
        n_particles,
        max_iterations: max_iter,
        archive_size,
        seed,
        mutation_rate,
        grid_divisions,
    };

    if let Some(names) = var_types {
        if names.len() != bounds.len() {
            return Err(PyValueError::new_err("var_types length must match bounds"));
        }
        let space = MixedSpace::new(bounds, build_var_types(&names)?);
        run_multi(py, space, vel, params, objective, false)
    } else if integer || binary {
        let int_bounds: Vec<(i64, i64)> = if binary {
            vec![(0, 1); bounds.len()]
        } else {
            bounds
                .iter()
                .map(|&(lo, hi)| (lo.round() as i64, hi.round() as i64))
                .collect()
        };
        run_multi(
            py,
            IntegerSpace::new(int_bounds),
            vel,
            params,
            objective,
            true,
        )
    } else {
        run_multi(
            py,
            ContinuousSpace::new(bounds),
            vel,
            params,
            objective,
            false,
        )
    }
}

/// Metadata of a native benchmark: `(bound, optimum_value)`. `bound` is the
/// recommended symmetric bound per dimension. Useful for auto-adjusting plot
/// domains or building `bounds` without hardcoding them by hand.
#[pyfunction]
fn benchmark_info(name: &str) -> PyResult<(f64, f64)> {
    match turboswarm_core::benchmarks::meta(name) {
        Some(b) => Ok((b.bound, b.optimum_value)),
        None => Err(PyKeyError::new_err(format!(
            "unknown native benchmark: '{name}'"
        ))),
    }
}

/// Metadata of a native grey benchmark: ``(center_bound, max_spread,
/// optimum_value)``. ``center_bound`` is the recommended symmetric bound for
/// each grey variable's center; ``max_spread`` the recommended maximum spread.
/// Useful for building ``bounds``/``max_spread`` without hardcoding them.
#[pyfunction]
fn grey_benchmark_info(name: &str) -> PyResult<(f64, f64, f64)> {
    match turboswarm_core::benchmarks::grey_meta(name) {
        Some(b) => Ok((b.center_bound, b.max_spread, b.optimum_value)),
        None => Err(PyKeyError::new_err(format!(
            "unknown native grey benchmark: '{name}'"
        ))),
    }
}

#[pymodule]
fn turboswarm_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_multi, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_grey, m)?)?;
    m.add_function(wrap_pyfunction!(py_hypervolume, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_info, m)?)?;
    m.add_function(wrap_pyfunction!(grey_benchmark_info, m)?)?;
    m.add_class::<PyPsoResult>()?;
    m.add_class::<PyParetoFront>()?;
    m.add_class::<PyGreyResult>()?;
    Ok(())
}
