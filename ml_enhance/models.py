from collections.abc import Callable
from typing import Any

from correlation_filter import CorrelationFilter
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

# TODO: Instead of using a general type, use a protocol
type ParamSearch = Callable[[BaseEstimator, Any, int], BaseEstimator]


def pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("remove_corr", CorrelationFilter(threshold=0.95)),
            ("transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
            ("predict", model),
        ]
    )


def make_random_search(
    estimator: BaseEstimator, param_dist: dict[str, Any] | list[dict[str, Any]], n_cpus: int
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=50,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        random_state=40,
        verbose=12,
        n_jobs=n_cpus,
    )


def setup_RF(n_cpus: int, hyperparam_opt: ParamSearch = make_random_search) -> BaseEstimator:
    pl_rf = pipeline(RandomForestRegressor(random_state=40))

    rf_param_dist: dict[str, Any] = {
        "predict__n_estimators": randint(200, 1000),
        "predict__max_depth": [None] + list(range(5, 40)),
        "predict__min_samples_split": randint(2, 20),
        "predict__min_samples_leaf": randint(1, 10),
        "predict__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }

    return hyperparam_opt(pl_rf, rf_param_dist, n_cpus=n_cpus)


def setup_KRR(n_cpus: int, hyperparam_opt: ParamSearch = make_random_search) -> BaseEstimator:
    pl_krr = pipeline(KernelRidge())

    krr_param_dist: list[dict[str, Any]] = [
        {"predict__kernel": ["rbf"], "predict__alpha": loguniform(1e-4, 1e2), "predict__gamma": loguniform(1e-4, 1e2)},
        {"predict__kernel": ["linear"], "predict__alpha": loguniform(1e-4, 1e2)},
        {
            "predict__kernel": ["polynomial"],
            "predict__alpha": loguniform(1e-4, 1e2),
            "predict__gamma": loguniform(1e-4, 1e2),
            "predict__degree": randint(2, 6),
            "predict__coef0": loguniform(1e-2, 1e2),
        },
    ]

    return hyperparam_opt(pl_krr, krr_param_dist, n_cpus=n_cpus)


def setup_Huber(n_cpus: int, hyperparam_opt: ParamSearch = make_random_search) -> BaseEstimator:
    pl_huber = pipeline(HuberRegressor(max_iter=1000))

    param_dist = {"predict__epsilon": uniform(1.0, 1.0), "predict__alpha": loguniform(1e-5, 1e-2)}

    return hyperparam_opt(pl_huber, param_dist, n_cpus=n_cpus)
