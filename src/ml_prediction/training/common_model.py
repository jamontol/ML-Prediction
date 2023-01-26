"""
TODO: Add docstrings (JAVIER)
"""

import os

import optuna
import pandas as pd
import xgboost as xgb

# from imblearn.ensemble import (
#     BalancedBaggingClassifier,
#     BalancedRandomForestClassifier,
#     EasyEnsembleClassifier,
#     RUSBoostClassifier,
# )
from lightgbm import LGBMClassifier, LGBMRegressor

# from requests import models
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, cross_val_score  # , StratifiedKFold

# from sklearn.svm import LinearSVC
from skopt import BayesSearchCV
from skopt.space import Categorical  # ,  Integer, Real

from ceramic import logger

from .model_params import get_model_params

# from mapie.quantile_regression import MapieQuantileRegressor
# from mapie.regression import MapieRegressor
# from .train_test import eval_model

log = logger.get_logger(__name__)


# TODO from mapie.regression import MapieRegressor  (homoscedastic)
# TODO from mapie.quantile_regression import MapieQuantileRegressor (heteroscedastic)

model_mode = os.getenv("MODE", "training")

# if model_mode == "training":
#     try:
#         import optuna
#         # import mlflow
#         print(f"Mode {model_mode}: Training libraries imported")
#     except ImportError:
#         print("No libraries installed")
# else:
#     print(f"Mode {model_mode}: Training libraries not imported")


def training_model(
    pipeline,
    features: pd.DataFrame,
    target,
    objective: str = "regression",
    scoring: str = "accuracy",
    test_size=0.2,
    n_trials=500,
    timeout=3600,
    HPO=True,
    interval_strategy: dict | None = None,
):
    """
    Full training pipeline for ML model with
    hyperparameter optimization using Bayesian method

    Parameters
    ----------
    pipeline: scikit Pipeline
    features: pd.DataFrame holding the model features
    target: pd.Series holding the target values
    params: dict holding the hyperparameters
    n_trials: maximum number of iterations in the optimization procedure

    Returns
    -------
    model: the optimized XGBoost model
    cv_score: the average RMSE coming from cross-validation
    test_score: the RMSE on the test set
    """

    if HPO == "Optuna":

        pipeline_params = {}

        if isinstance(pipeline, TransformedTargetRegressor):
            # pipeline_params = pipeline.get_params().keys()
            # print(f' PARAMS: {pipeline_params}')
            if "scaler" in pipeline.regressor.steps[0][0]:
                pipeline_params = {
                    "regressor__scaler__numerical__scaler_transformer__scaler_type": optuna.distributions.CategoricalDistribution(  # noqa=E501
                        ["min-max", "standard", "quantile_normal", "quantile_uniform"]
                    ),
                    "regressor__scaler__categorical__encoding_transformer__encoder_type": optuna.distributions.CategoricalDistribution(  # noqa=E501
                        ["one_hot", "ordinal"]
                    ),
                    "regressor__scaler__ordinal__encoding_transformer__encoder_type": optuna.distributions.CategoricalDistribution(  # noqa=E501
                        ["ordinal", "GLM"]
                    ),
                }
                model_params = get_model_params(pipeline.regressor.steps[-1][0], HPO)
                model_params = {"regressor__" + key: value for key, value in model_params.items()}
        else:
            if "scaler" in pipeline.steps[0][0]:
                pipeline_params = {
                    "scaler__numerical__scaler_transformer__scaler_type": optuna.distributions.CategoricalDistribution(
                        ["min-max", "standard", "quantile_normal", "quantile_uniform"]
                    ),
                    "scaler__categorical__encoding_transformer__encoder_type": optuna.distributions.CategoricalDistribution(  # noqa=E501
                        ["one_hot", "ordinal"]
                    ),
                    "scaler__ordinal__encoding_transformer__encoder_type": optuna.distributions.CategoricalDistribution(
                        ["ordinal", "GLM"]
                    ),
                }
                model_params = get_model_params(pipeline.steps[-1][0], HPO)

        params = {**pipeline_params, **model_params}

        if objective in ["binary", "multiclass", "multilabel"]:
            cv_space = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)  # StratifiedKFold(n_splits=5)
        else:  # regression
            cv_space = KFold(n_splits=5)

        # if estimator == 'sgd_regressor':
        #     enable_pruning = True
        # else:
        #     enable_pruning = False

        pipeline_cv = optuna.integration.OptunaSearchCV(
            pipeline, params, cv=cv_space, scoring=scoring, n_trials=n_trials, timeout=timeout
        )  # enable_pruning=enable_pruning,
        pipeline_cv.fit(features, target)
        best_pipeline = pipeline_cv.best_estimator_
        cv_score = pipeline_cv.best_score_
        best_params = pipeline_cv.best_params_

        print(f" {cv_score:.2f} {scoring} ")

        return best_pipeline, cv_score, best_params

    elif HPO == "GridSearch":
        pipeline_params = {}
        if isinstance(pipeline, TransformedTargetRegressor):
            if "scaler" in pipeline.regressor.steps[0][0]:
                pipeline_params = {
                    "regressor__scaler__numerical__scaler_transformer__scaler_type": [
                        "min-max",
                        "standard",
                        "quantile_normal",
                        "quantile_uniform",
                    ],
                    "regressor__scaler__categorical__encoding_transformer__encoder_type": ["one_hot", "ordinal", "GLM"],
                    "regressor__scaler__ordinal__encoding_transformer__encoder_type": ["ordinal", "GLM"]
                    # TODO include select_features
                }
                model_params = get_model_params(pipeline.regressor.steps[-1][0], HPO)
                model_params = {"regressor__" + key: value for key, value in model_params.items()}
        else:
            if "scaler" in pipeline.steps[0][0]:
                pipeline_params = {
                    "scaler__numerical__scaler_transformer__scaler_type": [
                        "min-max",
                        "standard",
                        "quantile_normal",
                        "quantile_uniform",
                    ],
                    "scaler__categorical__encoding_transformer__encoder_type": ["one_hot", "ordinal", "GLM"],
                    "scaler__ordinal__encoding_transformer__encoder_type": ["ordinal", "GLM"],
                }
                model_params = get_model_params(pipeline.steps[-1][0], HPO)

        params = {**pipeline_params, **model_params}

        if objective in ["binary", "multiclass", "multilabel"]:
            cv_space = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)  # StratifiedKFold(n_splits=5)
        else:  # regression
            cv_space = KFold(n_splits=5)

        pipeline_cv = GridSearchCV(pipeline, param_grid=params, cv=cv_space, scoring=scoring, verbose=2)

        pipeline_cv.fit(features, target)
        best_pipeline = pipeline_cv.best_estimator_

        # Non nested # TODO FIX: 'self' is not defined
        # if interval_strategy.keys() == "cqr":
        #     best_pipeline = MapieQuantileRegressor(best_pipeline, **self.interval_strategy.values())
        #     best_pipeline.fit(features, target)
        # elif interval_strategy.keys() in ["naive", "cv_plus", "jackknife_plus_ab"]:
        #     best_pipeline = MapieRegressor(best_pipeline, **self.interval_strategy.values())
        #     best_pipeline.fit(features, target)

        cv_score = pipeline_cv.best_score_
        best_params = pipeline_cv.best_params_

        print(f" {cv_score:.2f} {scoring}")

        return best_pipeline, cv_score, best_params

    elif HPO == "BayesSearch":

        pipeline_params = {}

        if isinstance(pipeline, TransformedTargetRegressor):
            if "scaler" in pipeline.regressor.steps[0][0]:
                pipeline_params = {
                    "regressor__scaler__numerical__scaler_transformer__scaler_type": Categorical(
                        ["min-max", "standard", "quantile_normal", "quantile_uniform"]
                    ),
                    "regressor__scaler__categorical__encoding_transformer__encoder_type": Categorical(
                        ["one_hot", "ordinal", "GLM"]
                    ),
                    "regressor__scaler__ordinal__encoding_transformer__encoder_type": Categorical(["ordinal", "GLM"]),
                }
                model_params = get_model_params(pipeline.regressor.steps[-1][0], HPO)
                model_params = {"regressor__" + key: value for key, value in model_params.items()}
        else:
            if "scaler" in pipeline.steps[0][0]:
                pipeline_params = {
                    "scaler__numerical__scaler_transformer__scaler_type": Categorical(
                        ["min-max", "standard", "quantile_normal", "quantile_uniform"]
                    ),
                    "scaler__categorical__encoding_transformer__encoder_type": Categorical(
                        ["one_hot", "ordinal", "GLM"]
                    ),
                    "scaler__ordinal__encoding_transformer__encoder_type": Categorical(["ordinal", "GLM"]),
                }
                model_params = get_model_params(pipeline.steps[-1][0], HPO)

        params = {**pipeline_params, **model_params}

        if objective in ["binary", "multiclass", "multilabel"]:
            cv_space = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)  # StratifiedKFold(n_splits=5)
        else:  # regression
            cv_space = KFold(n_splits=5)

        pipeline_cv = BayesSearchCV(pipeline, search_spaces=params, cv=cv_space, n_iter=10, scoring=scoring, verbose=2)
        pipeline_cv.fit(features, target)
        best_pipeline = pipeline_cv.best_estimator_
        cv_score = pipeline_cv.best_score_
        best_params = pipeline_cv.best_params_

        print(f" {cv_score.mean():.2f}")

        return best_pipeline, cv_score, best_params

    else:  # no HPO
        # eval_score=None

        if objective in ["binary", "multiclass", "multilabel"]:
            cv_space = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)  # StratifiedKFold(n_splits=5)
        else:  # regression
            cv_space = KFold(n_splits=3)

        cv_score = cross_val_score(pipeline, features, target, cv=cv_space, scoring=scoring)
        print(f" {cv_score.mean():.2f} {scoring} with a standard deviation of {cv_score.std():.2f}")

        pipeline.fit(features, target)
        eval_score = cv_score.mean()

        if isinstance(pipeline, TransformedTargetRegressor):
            best_params = pipeline.regressor.get_params()
        else:
            best_params = pipeline[-1].get_params()

        # eval_score = eval_model(pipeline, features, target)

        return pipeline, eval_score, best_params


def get_predictor(base_predictor: str, objective: str, params):

    if base_predictor == "xgboost":
        if objective == "regression":
            predictor = xgb.XGBRegressor(**params)
        else:
            predictor = xgb.XGBClassifier(**params)
    elif base_predictor == "random_forest":
        if objective == "regression":
            predictor = RandomForestRegressor(**params)
        else:
            predictor = RandomForestClassifier(**params)
    elif base_predictor == "extra_trees":
        if objective == "regression":
            predictor = ExtraTreesRegressor(**params)
        else:
            predictor = ExtraTreesClassifier(**params)
    elif base_predictor == "ligthgbm":
        if objective == "regression":
            predictor = LGBMRegressor(**params)
        else:
            predictor = LGBMClassifier(**params)
    elif base_predictor == "hist_gboost":
        if objective == "regression":
            predictor = HistGradientBoostingRegressor(**params)
        else:
            predictor = HistGradientBoostingClassifier(**params)
    elif base_predictor == "sgd":
        if objective == "regression":
            predictor = SGDRegressor(**params)
        else:
            predictor = SGDClassifier(**params)
    elif base_predictor == "quantile_regressor":
        if objective == "regression":
            predictor = SGDRegressor(**params)
        else:
            log.error(f"{base_predictor} is intended only for regression")

    return predictor
