//! Concrete test functions.

/// Metadata for a test function, useful for validation.
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// Name of the function (the same one used to dispatch it by string).
    pub name: &'static str,
    /// Recommended [min, max] bound per dimension (symmetric).
    pub bound: f64,
    /// Value of the global optimum.
    pub optimum_value: f64,
}

/// Sphere: f(x) = Σ xᵢ². Global minimum f(0) = 0. Convex, unimodal.
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Rastrigin: highly multimodal, global minimum f(0) = 0.
/// f(x) = 10n + Σ [xᵢ² − 10·cos(2π·xᵢ)]
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

/// Rosenbrock ("banana valley"): global minimum f(1,…,1) = 0.
/// f(x) = Σ [100·(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²]
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let (xi, xj) = (w[0], w[1]);
            100.0 * (xj - xi * xi).powi(2) + (1.0 - xi).powi(2)
        })
        .sum()
}

/// Ackley: multimodal, nearly flat away from the origin with a narrow well.
/// Global minimum f(0) = 0.
/// f(x) = −20·exp(−0.2·√(mean xᵢ²)) − exp(mean cos(2π·xᵢ)) + 20 + e
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x
        .iter()
        .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Griewank: a product of cosines that creates many regular local minima.
/// Global minimum f(0) = 0.
/// f(x) = 1 + Σ xᵢ²/4000 − Π cos(xᵢ/√i)
pub fn griewank(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / 4000.0;
    let prod: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
        .product();
    1.0 + sum - prod
}

/// Schwefel: multimodal and, unlike the others, with the optimum FAR from the
/// origin (at ≈420.97 per dimension). A good example of why centering
/// the search at 0 can be misleading. Global minimum f(420.9687…) = 0.
/// f(x) = 418.9829·n − Σ xᵢ·sin(√|xᵢ|)
pub fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.982_887_272_433_8 * n - x.iter().map(|&xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
}

/// Metadata for the Phase 1 functions.
pub const SPHERE: Benchmark = Benchmark {
    name: "sphere",
    bound: 5.12,
    optimum_value: 0.0,
};
pub const RASTRIGIN: Benchmark = Benchmark {
    name: "rastrigin",
    bound: 5.12,
    optimum_value: 0.0,
};
pub const ROSENBROCK: Benchmark = Benchmark {
    name: "rosenbrock",
    bound: 2.048,
    optimum_value: 0.0,
};

/// Metadata for the Phase 2 functions.
pub const ACKLEY: Benchmark = Benchmark {
    name: "ackley",
    bound: 32.768,
    optimum_value: 0.0,
};
pub const GRIEWANK: Benchmark = Benchmark {
    name: "griewank",
    bound: 600.0,
    optimum_value: 0.0,
};
pub const SCHWEFEL: Benchmark = Benchmark {
    name: "schwefel",
    bound: 500.0,
    optimum_value: 0.0,
};

/// All registered benchmarks with their metadata. It lets the
/// visualization layer choose bounds and know the optimum without
/// hardcoding them by hand (e.g. auto-fitting the domain of a plot).
pub const ALL: &[Benchmark] = &[SPHERE, RASTRIGIN, ROSENBROCK, ACKLEY, GRIEWANK, SCHWEFEL];

/// Looks up the metadata of a benchmark by name.
pub fn meta(name: &str) -> Option<&'static Benchmark> {
    ALL.iter().find(|b| b.name == name)
}
