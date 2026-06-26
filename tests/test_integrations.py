"""Tests for the optional library integrations (run with pytest)."""
import numpy as np
import pytest

import turboswarm as pso


# --- NumPy: bounds as arrays (issue #10) ------------------------------------
def test_numpy_bounds_matrix_matches_list_of_pairs():
    a = pso.minimize("sphere", bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]), seed=1)
    b = pso.minimize("sphere", bounds=[(-5.0, 5.0)] * 2, seed=1)
    assert a.best_position == b.best_position


def test_numpy_single_pair_with_dim():
    r = pso.minimize("sphere", bounds=np.array([-5.0, 5.0]), dim=3, seed=1)
    assert len(r.best_position) == 3
    assert r.best_value < 1e-2


def test_list_of_lists_bounds():
    r = pso.minimize("sphere", bounds=[[-5.0, 5.0], [-5.0, 5.0]], seed=1)
    assert len(r.best_position) == 2


def test_malformed_bound_row_errors():
    with pytest.raises(ValueError):
        pso.minimize("sphere", bounds=[[-5.0, 5.0, 1.0], [-5.0, 5.0, 1.0]], seed=1)


# --- SciPy wrapper (issue #11) ----------------------------------------------
def test_scipy_wrapper_returns_optimizeresult():
    pytest.importorskip("scipy")
    from scipy.optimize import OptimizeResult

    from turboswarm.integrations import scipy as ts_scipy

    def sphere(x):
        return float(np.sum(x ** 2))

    res = ts_scipy.minimize(sphere, bounds=[(-5, 5)] * 3, seed=0,
                            n_particles=40, max_iter=200)
    assert isinstance(res, OptimizeResult)
    assert res.x.shape == (3,)
    assert res.fun < 1e-2
    assert res.nit == 200
    assert res.nfev > 0
    assert res.success is True
    assert "max_iterations" in res.message


def test_scipy_wrapper_infers_dim_from_x0():
    pytest.importorskip("scipy")
    from turboswarm.integrations import scipy as ts_scipy

    res = ts_scipy.minimize(lambda x: float(np.sum(x ** 2)),
                            x0=[0.0, 0.0, 0.0], bounds=(-5, 5), seed=0)
    assert res.x.shape == (3,)


def test_scipy_wrapper_accepts_bounds_object_and_options():
    pytest.importorskip("scipy")
    from scipy.optimize import Bounds

    from turboswarm.integrations import scipy as ts_scipy

    res = ts_scipy.minimize(lambda x: float(np.sum(x ** 2)),
                            bounds=Bounds([-5, -5], [5, 5]),
                            options={"maxiter": 50}, seed=0)
    assert res.nit == 50
    assert res.x.shape == (2,)


def test_scipy_wrapper_requires_bounds():
    pytest.importorskip("scipy")
    from turboswarm.integrations import scipy as ts_scipy

    with pytest.raises(ValueError):
        ts_scipy.minimize(lambda x: float(np.sum(x ** 2)))


# --- Pandas export (issue #15) ----------------------------------------------
def test_pandas_convergence_dataframe():
    pytest.importorskip("pandas")
    from turboswarm.integrations import pandas as ts_pandas

    r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1, max_iter=10)
    df = ts_pandas.convergence_dataframe(r)
    assert list(df.columns) == ["iteration", "best_value"]
    assert len(df) == 10
    # convergence is monotonically non-increasing for the global best
    assert (df["best_value"].diff().dropna() <= 1e-9).all()


def test_pandas_history_dataframe_shape():
    pytest.importorskip("pandas")
    from turboswarm.integrations import pandas as ts_pandas

    r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1,
                     max_iter=10, n_particles=5)
    df = ts_pandas.history_dataframe(r)
    assert list(df.columns) == ["iteration", "particle", "x0", "x1"]
    assert len(df) == 10 * 5


def test_pandas_history_dataframe_requires_history():
    pytest.importorskip("pandas")
    from turboswarm.integrations import pandas as ts_pandas

    r = pso.minimize("sphere", bounds=[(-5, 5)] * 2, seed=1,
                     max_iter=5, record_history=False)
    with pytest.raises(ValueError):
        ts_pandas.history_dataframe(r)


# --- Joblib parallel objective (issue #16) ----------------------------------
def test_joblib_objective_converges():
    pytest.importorskip("joblib")
    from turboswarm.integrations import parallel

    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))

    obj = parallel.joblib_objective(sphere, n_jobs=2, backend="threading")
    r = pso.minimize(obj, bounds=[(-5, 5)] * 4, vectorized=True, seed=1)
    assert r.best_value < 1e-3


# --- scikit-learn PSOSearchCV (issue #13) -----------------------------------
def test_psosearchcv_tunes_and_predicts():
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    from turboswarm.integrations.sklearn import PSOSearchCV

    X, y = load_iris(return_X_y=True)
    search = PSOSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": (1, 8), "min_samples_leaf": (1, 10),
         "criterion": ["gini", "entropy"]},
        n_particles=8, max_iter=6, cv=3, random_state=0,
    )
    search.fit(X, y)

    assert search.best_score_ > 0.8
    assert set(search.best_params_) == {"max_depth", "min_samples_leaf", "criterion"}
    # integer/categorical decoding lands inside the declared space
    assert 1 <= search.best_params_["max_depth"] <= 8
    assert search.best_params_["criterion"] in {"gini", "entropy"}
    assert search.cv_results_["rank_test_score"].min() == 1
    assert search.predict(X[:5]).shape == (5,)
    assert 0.0 <= search.score(X, y) <= 1.0


def test_psosearchcv_clonable_and_respects_refit_false():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    from turboswarm.integrations.sklearn import PSOSearchCV

    X, y = load_iris(return_X_y=True)
    search = PSOSearchCV(
        LogisticRegression(max_iter=200),
        {"C": (1e-2, 1e2)},
        n_particles=6, max_iter=4, cv=3, random_state=0, refit=False,
    )
    # sklearn-cloneable (get_params/set_params round-trip via BaseEstimator)
    assert isinstance(clone(search), PSOSearchCV)
    search.fit(X, y)
    with pytest.raises(AttributeError):
        search.predict(X)


# --- Optuna sampler (issue #14) ---------------------------------------------
def test_turboswarm_sampler_minimizes():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from turboswarm.integrations.optuna import TurboswarmSampler

    def objective(trial):
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        sign = trial.suggest_categorical("sign", ["pos", "neg"])
        base = (x - 2) ** 2 + (y + 3) ** 2
        return base if sign == "pos" else base + 1.0

    study = optuna.create_study(direction="minimize",
                                sampler=TurboswarmSampler(n_particles=20, seed=0))
    study.optimize(objective, n_trials=200)
    assert study.best_value < 0.1
    assert study.best_params["sign"] == "pos"
    assert abs(study.best_params["x"] - 2) < 0.5
    assert abs(study.best_params["y"] + 3) < 0.5


def test_turboswarm_sampler_log_and_int_and_maximize():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from turboswarm.integrations.optuna import TurboswarmSampler

    def objective(trial):  # maximize: peak at n == 7
        n = trial.suggest_int("n", 1, 20)
        lr = trial.suggest_float("lr", 1e-4, 1e0, log=True)  # log-scale
        return -((n - 7) ** 2) - (lr - 0.1) ** 2

    study = optuna.create_study(direction="maximize",
                                sampler=TurboswarmSampler(n_particles=15, seed=1))
    study.optimize(objective, n_trials=150)
    assert abs(study.best_params["n"] - 7) <= 1


# --- Agents: LangChain tool (issue #17) -------------------------------------
def test_optimization_tool_invokes_pso():
    pytest.importorskip("langchain_core")
    from turboswarm.integrations.agents import optimization_tool

    tool = optimization_tool()
    assert tool.name == "minimize_benchmark"
    assert set(tool.args) >= {"benchmark", "dim", "bound"}

    out = tool.invoke({"benchmark": "rastrigin", "dim": 2, "n_particles": 40,
                       "max_iter": 150, "seed": 0})
    assert out["best_value"] < 1e-2
    assert len(out["best_position"]) == 2
    assert out["stop_reason"] == "max_iterations"


def test_optimization_tool_rejects_unknown_benchmark():
    pytest.importorskip("langchain_core")
    from turboswarm.integrations.agents import optimization_tool

    tool = optimization_tool()
    with pytest.raises(Exception):
        tool.invoke({"benchmark": "definitely_not_a_benchmark", "dim": 2})


# --- PyTorch gradient-free example (issue #12) ------------------------------
def test_pytorch_gradient_free_training():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, size=(200, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    shapes = [p.shape for p in model.parameters()]
    sizes = [p.numel() for p in model.parameters()]

    def set_params(vec):
        i = 0
        with torch.no_grad():
            for p, shape, size in zip(model.parameters(), shapes, sizes):
                p.copy_(torch.tensor(vec[i:i + size],
                                     dtype=torch.float32).view(shape))
                i += size

    def neg_accuracy(vec):
        set_params(vec)
        with torch.no_grad():
            pred = model(Xt).argmax(dim=1)
            return -(pred == yt).float().mean().item()

    result = pso.minimize(neg_accuracy, bounds=[(-3, 3)] * sum(sizes),
                          n_particles=50, max_iter=150, seed=0)
    assert -result.best_value > 0.9      # gradient-free, non-differentiable metric
