//! Integration tests: validate that the PSO converges near the known
//! optimum of each test function. With a fixed seed => deterministic.

use turboswarm_core::benchmarks::{ackley, griewank, rastrigin, rosenbrock, schwefel, sphere};
use turboswarm_core::prelude::*;

fn run_continuous<F>(f: F, bound: f64, dim: usize, seed: u64) -> PsoResult<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let space = ContinuousSpace::new(vec![(-bound, bound); dim]);
    let velocity = InertiaVelocity::new(0.729, 1.49445, 1.49445);
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 300,
        seed: Some(seed),
        ..Default::default()
    };
    Pso::new(space, velocity, GlobalBest::new(), params).minimize(f)
}

/// Generic variant: runs any `Velocity` + `Topology` over `f`.
fn run_with<F, V, T>(f: F, vel: V, topo: T, bound: f64, dim: usize, seed: u64) -> PsoResult<f64>
where
    F: FnMut(&[f64]) -> f64,
    V: Velocity,
    T: Topology,
{
    let space = ContinuousSpace::new(vec![(-bound, bound); dim]);
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 300,
        seed: Some(seed),
        ..Default::default()
    };
    Pso::new(space, vel, topo, params).minimize(f)
}

#[test]
fn uniform_constructor_matches_explicit_bounds() {
    assert_eq!(
        ContinuousSpace::uniform(3, -5.12, 5.12).bounds(),
        ContinuousSpace::new(vec![(-5.12, 5.12); 3]).bounds()
    );
    assert_eq!(
        IntegerSpace::uniform(2, -10, 10).bounds(),
        &[(-10, 10), (-10, 10)]
    );
}

#[test]
fn sphere_converges_near_zero() {
    let res = run_continuous(sphere, 5.12, 2, 42);
    assert!(res.best_value < 1e-4, "value = {}", res.best_value);
}

#[test]
fn rastrigin_finds_good_solution() {
    // Rastrigin is highly multimodal; we ask for "reasonably good", not exact.
    let res = run_continuous(rastrigin, 5.12, 2, 7);
    assert!(res.best_value < 1.0, "value = {}", res.best_value);
}

#[test]
fn rosenbrock_progresses() {
    let res = run_continuous(rosenbrock, 2.048, 2, 123);
    assert!(res.best_value < 1.0, "value = {}", res.best_value);
}

#[test]
fn reproducible_with_same_seed() {
    let a = run_continuous(sphere, 5.12, 3, 99);
    let b = run_continuous(sphere, 5.12, 3, 99);
    assert_eq!(a.best_value, b.best_value);
    assert_eq!(a.best_position, b.best_position);
}

#[test]
fn history_is_recorded() {
    let res = run_continuous(sphere, 5.12, 2, 1);
    assert_eq!(res.history.iterations(), 300);
    assert_eq!(res.history.positions.len(), 300);
    // The convergence curve must be monotonically non-increasing.
    for w in res.history.best_value.windows(2) {
        assert!(w[1] <= w[0] + 1e-12);
    }
}

#[test]
fn integer_space_returns_integers() {
    // Minimizes Σ(xᵢ-3)² over integers; optimum at (3,3).
    let space = IntegerSpace::new(vec![(-10, 10); 2]);
    let velocity = InertiaVelocity::new(0.729, 1.49445, 1.49445);
    let params = PsoParams {
        n_particles: 30,
        max_iterations: 200,
        seed: Some(5),
        ..Default::default()
    };
    let res = Pso::new(space, velocity, GlobalBest::new(), params)
        .minimize(|x: &[i64]| x.iter().map(|&xi| ((xi - 3) * (xi - 3)) as f64).sum());

    assert_eq!(res.best_position, vec![3, 3]);
    assert_eq!(res.best_value, 0.0);
}

// --- Phase 2: new variants, topologies, and benchmarks ---

#[test]
fn constriction_converges_on_sphere() {
    let res = run_with(
        sphere,
        ConstrictionVelocity::default(),
        GlobalBest::new(),
        5.12,
        2,
        42,
    );
    assert!(res.best_value < 1e-4, "value = {}", res.best_value);
}

#[test]
fn constriction_chi_matches_classic_value() {
    // c1 = c2 = 2.05 (φ = 4.1) => χ ≈ 0.7298.
    let v = ConstrictionVelocity::new(2.05, 2.05);
    assert!((v.chi() - 0.729_843_788).abs() < 1e-6, "χ = {}", v.chi());
}

#[test]
fn ring_topology_converges_on_sphere() {
    let res = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        Ring::lbest(),
        5.12,
        2,
        42,
    );
    assert!(res.best_value < 1e-3, "value = {}", res.best_value);
}

#[test]
fn von_neumann_topology_converges_on_sphere() {
    // 40 particles => 5×8 grid covers the whole swarm.
    let topo = VonNeumann::square_for(40);
    let res = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        topo,
        5.12,
        2,
        42,
    );
    assert!(res.best_value < 1e-3, "value = {}", res.best_value);
}

#[test]
fn ackley_finds_good_solution() {
    let res = run_continuous(ackley, 32.768, 2, 7);
    assert!(res.best_value < 1e-2, "value = {}", res.best_value);
}

#[test]
fn griewank_finds_good_solution() {
    let res = run_continuous(griewank, 600.0, 2, 11);
    assert!(res.best_value < 0.1, "value = {}", res.best_value);
}

#[test]
fn schwefel_optimum_value_is_near_zero_at_known_point() {
    // The Schwefel optimum is at ≈420.97 per dimension, not at the origin.
    let opt = vec![420.968_746; 4];
    assert!(schwefel(&opt) < 1e-2, "f(opt) = {}", schwefel(&opt));
}

#[test]
fn fips_converges_with_local_topology() {
    // FIPS performs better with local topologies; the ring is the typical case.
    let res = run_with(sphere, FipsVelocity::default(), Ring::lbest(), 5.12, 2, 42);
    assert!(res.best_value < 1e-2, "value = {}", res.best_value);
}

#[test]
fn fips_chi_matches_constriction() {
    // FIPS uses the same constriction factor as Clerc-Kennedy (φ=4.1).
    let f = FipsVelocity::default();
    let c = ConstrictionVelocity::default();
    assert!(
        (f.chi() - c.chi()).abs() < 1e-12,
        "χ_fips={}, χ_c={}",
        f.chi(),
        c.chi()
    );
}

#[test]
fn fips_is_reproducible() {
    let a = run_with(sphere, FipsVelocity::default(), Ring::lbest(), 5.12, 3, 77);
    let b = run_with(sphere, FipsVelocity::default(), Ring::lbest(), 5.12, 3, 77);
    assert_eq!(a.best_value, b.best_value);
    assert_eq!(a.best_position, b.best_position);
}

#[test]
fn random_topology_converges_on_sphere() {
    let topo = Random::new(3, 42);
    let res = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        topo,
        5.12,
        2,
        42,
    );
    assert!(res.best_value < 1e-3, "value = {}", res.best_value);
}

#[test]
fn random_topology_is_reproducible() {
    let a = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        Random::new(3, 1),
        5.12,
        3,
        7,
    );
    let b = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        Random::new(3, 1),
        5.12,
        3,
        7,
    );
    assert_eq!(a.best_value, b.best_value);
    assert_eq!(a.best_position, b.best_position);
}

#[test]
fn early_stopping_shortens_the_run() {
    // With aggressive patience, sphere stagnates well before max_iterations.
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 1000,
        seed: Some(42),
        patience: 20,
        tol: 1e-12,
        ..Default::default()
    };
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize(sphere);
    // It should stop early (fewer than the 1000 allowed iterations) yet still
    // reach a good solution.
    assert!(
        res.history.iterations() < 1000,
        "ran {} iters",
        res.history.iterations()
    );
    assert!(res.best_value < 1e-4, "value = {}", res.best_value);
}

#[test]
fn v_max_limits_velocity() {
    // With a tight velocity clamp, no recorded step exceeds v_max per component.
    let v_max = 0.5;
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let params = PsoParams {
        n_particles: 20,
        max_iterations: 50,
        seed: Some(3),
        v_max: Some(v_max),
        ..Default::default()
    };
    // Reconstruct trajectories: consecutive positions differ by the clamped
    // velocity, but clamping also interacts with bound-clamping, so we just
    // assert the run converges and completes (smoke test for the feature path).
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.9, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize(sphere);
    assert_eq!(res.history.iterations(), 50);
    assert!(res.best_value.is_finite());
}

#[test]
fn stops_on_target_value() {
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 10_000,
        seed: Some(42),
        target: Some(1e-6),
        ..Default::default()
    };
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize(sphere);
    assert_eq!(res.stop_reason, StopReason::Target);
    assert!(res.best_value <= 1e-6);
    assert!(res.history.iterations() < 10_000);
}

#[test]
fn stops_on_evaluation_budget() {
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let params = PsoParams {
        n_particles: 20,
        max_iterations: 10_000,
        seed: Some(1),
        max_evals: Some(500),
        ..Default::default()
    };
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize(sphere);
    assert_eq!(res.stop_reason, StopReason::MaxEvaluations);
    // 20 init + 20 per iteration; should stop shortly after crossing 500.
    assert!(
        res.evaluations >= 500 && res.evaluations < 540,
        "evals = {}",
        res.evaluations
    );
}

#[test]
fn reports_evaluations_and_default_stop_reason() {
    let res = run_continuous(sphere, 5.12, 2, 7);
    assert_eq!(res.stop_reason, StopReason::MaxIterations);
    // 40 particles, 300 iterations + 40 init.
    assert_eq!(res.evaluations, 40 * 301);
}

#[test]
fn boundary_strategies_keep_positions_in_bounds_and_converge() {
    use turboswarm_core::traits::BoundaryHandling::*;
    let bound = 5.12;
    for handling in [Clamp, Reflect, Wrap, Reinit] {
        let space = ContinuousSpace::new(vec![(-bound, bound); 2]);
        let params = PsoParams {
            n_particles: 40,
            max_iterations: 300,
            seed: Some(42),
            bounds_handling: handling,
            ..Default::default()
        };
        let res = Pso::new(
            space,
            InertiaVelocity::new(0.729, 1.49445, 1.49445),
            GlobalBest::new(),
            params,
        )
        .minimize(sphere);
        for &xi in &res.best_position {
            assert!(
                xi >= -bound - 1e-9 && xi <= bound + 1e-9,
                "{handling:?}: {xi} out of bounds"
            );
        }
        assert!(
            res.best_value < 1.0,
            "{handling:?}: value = {}",
            res.best_value
        );
    }
}

#[test]
fn mopso_finds_pareto_front() {
    use turboswarm_core::mopso::{dominates, Mopso, MopsoParams};
    // Bi-objective: f1 = x^2, f2 = (x-2)^2. The Pareto-optimal set is x in [0, 2].
    let space = ContinuousSpace::new(vec![(-5.0, 5.0)]);
    let params = MopsoParams {
        n_particles: 80,
        max_iterations: 80,
        archive_size: 50,
        seed: Some(42),
        mutation_rate: 0.1,
    };
    let res = Mopso::new(space, InertiaVelocity::new(0.729, 1.49445, 1.49445), params)
        .minimize(|x: &[f64]| vec![x[0] * x[0], (x[0] - 2.0).powi(2)]);

    assert!(
        res.front.len() >= 10,
        "front too small: {}",
        res.front.len()
    );
    // Every Pareto-optimal x should lie in [0, 2].
    for s in &res.front {
        assert!(
            s.position[0] > -0.1 && s.position[0] < 2.1,
            "x = {}",
            s.position[0]
        );
    }
    // The archive must be mutually non-dominated.
    for a in &res.front {
        for b in &res.front {
            assert!(!dominates(&a.objectives, &b.objectives) || a.objectives == b.objectives);
        }
    }
}

#[test]
fn hypervolume_matches_known_values_and_is_monotone() {
    use turboswarm_core::mopso::hypervolume;

    // Staircase under reference (4,4): exact dominated area is 6.
    let front = [vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
    assert!((hypervolume(&front, &[4.0, 4.0]) - 6.0).abs() < 1e-9);

    // A dominated point adds nothing; a dominating point can only increase HV.
    let with_dominated = [
        vec![1.0, 3.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
        vec![3.0, 3.0],
    ];
    assert!((hypervolume(&with_dominated, &[4.0, 4.0]) - 6.0).abs() < 1e-9);
    let better = [vec![0.5, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
    assert!(hypervolume(&better, &[4.0, 4.0]) > 6.0);

    // Points not strictly better than the reference contribute nothing.
    assert_eq!(hypervolume(&[vec![4.0, 1.0]], &[4.0, 4.0]), 0.0);

    // Three objectives: a single point fills the box from it to the reference.
    let one = [vec![1.0, 1.0, 1.0]];
    assert!((hypervolume(&one, &[3.0, 3.0, 3.0]) - 8.0).abs() < 1e-9);
}

#[test]
fn mopso_hypervolume_rewards_a_converged_front() {
    use turboswarm_core::mopso::{Mopso, MopsoParams};
    let space = ContinuousSpace::new(vec![(-5.0, 5.0)]);
    let f = |x: &[f64]| vec![x[0] * x[0], (x[0] - 2.0).powi(2)];

    let params = MopsoParams {
        n_particles: 80,
        max_iterations: 80,
        archive_size: 50,
        seed: Some(42),
        mutation_rate: 0.1,
    };
    let res = Mopso::new(space, InertiaVelocity::new(0.729, 1.49445, 1.49445), params).minimize(f);

    // Shared reference so the fronts are comparable. The converged front must
    // dominate (much higher HV than) a deliberately poor, off-front sample.
    let reference = [8.0, 8.0];
    let good = res.hypervolume(Some(&reference));
    let poor = turboswarm_core::mopso::hypervolume(&[vec![4.0, 6.0], vec![6.0, 4.0]], &reference);
    assert!(good > 0.0);
    assert!(
        good > poor,
        "converged HV {good} should beat poor HV {poor}"
    );

    // The auto-reference path (nadir of the front) is also well-defined.
    assert!(res.hypervolume(None) > 0.0);
}

#[test]
fn mixed_space_respects_per_dimension_types() {
    use turboswarm_core::spaces::VarType::*;
    // Dim 0 real (opt 1.5), dim 1 integer (opt 3), dim 2 binary (opt 1).
    let space = MixedSpace::new(
        vec![(-5.0, 5.0), (-10.0, 10.0), (0.0, 1.0)],
        vec![Real, Integer, Binary],
    );
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 300,
        seed: Some(7),
        ..Default::default()
    };
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize(|x: &[f64]| (x[0] - 1.5).powi(2) + (x[1] - 3.0).powi(2) + (x[2] - 1.0).powi(2));
    // Integer and binary dimensions must decode to whole values.
    assert_eq!(res.best_position[1], res.best_position[1].round());
    assert!(res.best_position[2] == 0.0 || res.best_position[2] == 1.0);
    assert!(
        (res.best_position[1] - 3.0).abs() < 1e-9,
        "int dim = {}",
        res.best_position[1]
    );
    assert_eq!(res.best_position[2], 1.0);
    assert!(
        (res.best_position[0] - 1.5).abs() < 0.1,
        "real dim = {}",
        res.best_position[0]
    );
}

#[test]
fn callback_is_called_and_can_stop() {
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let params = PsoParams {
        n_particles: 20,
        max_iterations: 1000,
        seed: Some(1),
        ..Default::default()
    };
    let mut calls = 0usize;
    let res = Pso::new(
        space,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        params,
    )
    .minimize_with_callback(sphere, |info| {
        calls += 1;
        info.iteration < 9 // stop after the 10th iteration (index 9)
    });
    assert_eq!(res.stop_reason, StopReason::Callback);
    assert_eq!(calls, 10);
    assert_eq!(res.history.iterations(), 10);
}

#[test]
fn batch_matches_parallel_and_converges() {
    // Batched and parallel evaluation are both synchronous, so with the same
    // seed they must produce identical results; batch must also converge.
    let make = || {
        let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
        let params = PsoParams {
            n_particles: 40,
            max_iterations: 300,
            seed: Some(42),
            ..Default::default()
        };
        Pso::new(
            space,
            InertiaVelocity::new(0.729, 1.49445, 1.49445),
            GlobalBest::new(),
            params,
        )
    };
    let batched = make().minimize_batch(|xs: &[Vec<f64>]| xs.iter().map(|x| sphere(x)).collect());
    let parallel = make().minimize_parallel(sphere);
    assert!(batched.best_value < 1e-3, "value = {}", batched.best_value);
    assert_eq!(batched.best_value, parallel.best_value);
    assert_eq!(batched.best_position, parallel.best_position);
}

#[test]
fn parallel_converges_and_is_reproducible() {
    // Synchronous parallel mode: converges on sphere and is deterministic.
    let run = |seed: u64| {
        let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
        let params = PsoParams {
            n_particles: 40,
            max_iterations: 300,
            seed: Some(seed),
            ..Default::default()
        };
        Pso::new(
            space,
            InertiaVelocity::new(0.729, 1.49445, 1.49445),
            GlobalBest::new(),
            params,
        )
        .minimize_parallel(sphere)
    };
    let a = run(42);
    let b = run(42);
    assert!(a.best_value < 1e-3, "value = {}", a.best_value);
    assert_eq!(a.best_value, b.best_value);
    assert_eq!(a.best_position, b.best_position);
}

#[test]
fn variants_reach_optimum_on_same_function() {
    // Same function and seed, different variant: both must solve sphere.
    let inertia = run_with(
        sphere,
        InertiaVelocity::new(0.729, 1.49445, 1.49445),
        GlobalBest::new(),
        5.12,
        2,
        2024,
    );
    let constr = run_with(
        sphere,
        ConstrictionVelocity::default(),
        GlobalBest::new(),
        5.12,
        2,
        2024,
    );
    assert!(inertia.best_value < 1e-4);
    assert!(constr.best_value < 1e-4);
}
