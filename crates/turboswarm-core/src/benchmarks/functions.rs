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

// --- CEC-family functions ---
//
// These are the canonical (unshifted, unrotated) base functions used as the
// building blocks of the CEC benchmark suites. The official suites compose them
// with shift vectors and rotation matrices supplied as data files; those are
// not bundled here, but a shift/rotation can be applied by the caller. All have
// their global minimum equal to 0.

/// Bent Cigar: unimodal and severely ill-conditioned — one cheap direction and
/// the rest scaled by 10⁶. Global minimum f(0) = 0.
/// f(x) = x₁² + 10⁶·Σ_{i>1} xᵢ²
pub fn bent_cigar(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x[0] * x[0] + 1e6 * x[1..].iter().map(|&xi| xi * xi).sum::<f64>()
}

/// Discus: the ill-conditioned counterpart of Bent Cigar — one expensive
/// direction (10⁶) and the rest cheap. Global minimum f(0) = 0.
/// f(x) = 10⁶·x₁² + Σ_{i>1} xᵢ²
pub fn discus(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    1e6 * x[0] * x[0] + x[1..].iter().map(|&xi| xi * xi).sum::<f64>()
}

/// High Conditioned Elliptic: a smoothly increasing condition number across
/// dimensions (from 1 to 10⁶). Unimodal. Global minimum f(0) = 0.
/// f(x) = Σᵢ (10⁶)^((i−1)/(n−1))·xᵢ²
pub fn elliptic(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return x.first().map_or(0.0, |&v| v * v);
    }
    x.iter()
        .enumerate()
        .map(|(i, &xi)| 1e6_f64.powf(i as f64 / (n - 1) as f64) * xi * xi)
        .sum()
}

/// Zakharov: unimodal with no local minima, coupling the dimensions through a
/// weighted sum. Global minimum f(0) = 0.
/// f(x) = Σxᵢ² + (Σ 0.5·i·xᵢ)² + (Σ 0.5·i·xᵢ)⁴
pub fn zakharov(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_half: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| 0.5 * (i + 1) as f64 * xi)
        .sum();
    sum_sq + sum_half.powi(2) + sum_half.powi(4)
}

/// Levy: multimodal with many local minima. Global minimum f(1,…,1) = 0
/// (the optimum is away from the origin).
pub fn levy(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();
    let term1 = (pi * w[0]).sin().powi(2);
    let term_mid: f64 = w[..n - 1]
        .iter()
        .map(|&wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (pi * wi + 1.0).sin().powi(2)))
        .sum();
    let term_last = (w[n - 1] - 1.0).powi(2) * (1.0 + (2.0 * pi * w[n - 1]).sin().powi(2));
    term1 + term_mid + term_last
}

/// Expanded Schaffer F6: a deceptive, highly multimodal function built by
/// chaining the 2-D Schaffer F6 over consecutive (cyclic) pairs. Global
/// minimum f(0) = 0.
pub fn expanded_schaffer(x: &[f64]) -> f64 {
    fn g(a: f64, b: f64) -> f64 {
        let s = a * a + b * b;
        0.5 + (s.sqrt().sin().powi(2) - 0.5) / (1.0 + 0.001 * s).powi(2)
    }
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return g(x[0], x[0]);
    }
    (0..n).map(|i| g(x[i], x[(i + 1) % n])).sum()
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

/// Metadata for the CEC-family functions.
pub const BENT_CIGAR: Benchmark = Benchmark {
    name: "bent_cigar",
    bound: 100.0,
    optimum_value: 0.0,
};
pub const DISCUS: Benchmark = Benchmark {
    name: "discus",
    bound: 100.0,
    optimum_value: 0.0,
};
pub const ELLIPTIC: Benchmark = Benchmark {
    name: "elliptic",
    bound: 100.0,
    optimum_value: 0.0,
};
pub const ZAKHAROV: Benchmark = Benchmark {
    name: "zakharov",
    bound: 10.0,
    optimum_value: 0.0,
};
pub const LEVY: Benchmark = Benchmark {
    name: "levy",
    bound: 10.0,
    optimum_value: 0.0,
};
pub const EXPANDED_SCHAFFER: Benchmark = Benchmark {
    name: "expanded_schaffer",
    bound: 100.0,
    optimum_value: 0.0,
};

/// All registered benchmarks with their metadata. It lets the
/// visualization layer choose bounds and know the optimum without
/// hardcoding them by hand (e.g. auto-fitting the domain of a plot).
pub const ALL: &[Benchmark] = &[
    SPHERE,
    RASTRIGIN,
    ROSENBROCK,
    ACKLEY,
    GRIEWANK,
    SCHWEFEL,
    BENT_CIGAR,
    DISCUS,
    ELLIPTIC,
    ZAKHAROV,
    LEVY,
    EXPANDED_SCHAFFER,
];

/// Looks up the metadata of a benchmark by name.
pub fn meta(name: &str) -> Option<&'static Benchmark> {
    ALL.iter().find(|b| b.name == name)
}
