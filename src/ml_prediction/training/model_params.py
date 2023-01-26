# """
# Javier Monreal Tolmo
# TODO: Add docstrings (JAVIER)
# """

import optuna
from skopt.space import Categorical, Integer, Real

params_LGM_OPT = {
    # "metric": "RMSE"
    # 'device_type ': 'gpu',
    "ligthgbm__boosting_type": optuna.distributions.CategoricalDistribution(["gbdt"]),  # 'rf', 'dart'
    # 'ligthgbm__reg_alpha': optuna.distributions.IntDistribution(1e-8, 1),
    # 'ligthgbm__reg_lambda': optuna.distributions.IntDistribution(1e-8, 1),
    "ligthgbm__learning_rate": optuna.distributions.FloatDistribution(1e-2, 1),
    "ligthgbm__max_depth": optuna.distributions.IntDistribution(5, 30, 1),
    "ligthgbm__num_leaves": optuna.distributions.IntDistribution(2, 256),
    # 'ligthgbm__colsample_bytree': optuna.distributions.UniformDistribution(0.4, 1.0),
    # 'ligthgbm__subsample': optuna.distributions.UniformDistribution(0.4, 1.0),
    # 'ligthgbm__subsample_freq': optuna.distributions.IntUniformDistribution(1, 70),
    "ligthgbm__min_child_samples": optuna.distributions.IntDistribution(5, 100),
    # 'ligthgbm__seed':2018
}

params_LGM_BS = {
    "ligthgbm__boosting_type": Categorical(["gbdt"]),  # 'rf', 'dart'
    # 'ligthgbm__reg_alpha': Real(1e-8, 1e+0, prior='log-uniform'),
    # 'ligthgbm__reg_lambda': Real(1e-8, 1e+0, prior='log-uniform'),
    "ligthgbm__learning_rate": Real(1e-2, 1e0, prior="log-uniform"),
    "ligthgbm__max_depth": Integer(5, 30),
    "ligthgbm__num_leaves": Integer(2, 256),
    # 'ligthgbm__colsample_bytree': Real(0.4, 1.0, prior='uniform'),
    # 'ligthgbm__subsample': Real(0.4, 1.0, prior='uniform'),
    # 'ligthgbm__subsample_freq': Integer(1, 70),
    "ligthgbm__min_child_samples": Integer(5, 100),
    # 'ligthgbm__seed':[2018]
}


params_XGB_OPT = {
    "xgboost__eta": optuna.distributions.FloatDistribution(1e-2, 1),
    "xgboost__max_depth": optuna.distributions.IntDistribution(5, 30, 1),
    "xgboost__min_child_weight": optuna.distributions.IntDistribution(1, 10),
    "xgboost__n_estimators": optuna.distributions.IntDistribution(10, 100, 10),
}

params_LGM_GS = {
    "ligthgbm__boosting_type": ["gbdt"],  # 'rf', 'dart'
    # 'ligthgbm__reg_alpha': [0.0001, 1],
    # 'ligthgbm__reg_lambda': [ 0.001, 1],
    "ligthgbm__learning_rate": [0.01, 1],
    "ligthgbm__max_depth": [5, 30],
    "ligthgbm__num_leaves": [2, 256],
    # 'ligthgbm__colsample_bytree': [0.4, 1.0],
    # 'ligthgbm__subsample': [0.4, 1.0],
    # 'ligthgbm__subsample_freq': [ 1, 70],
    "ligthgbm__min_child_samples": [5, 100],
    # 'ligthgbm__seed':[2018]
}

params_RF_OPT = {
    # 'random_forest__bootstrap': optuna.distributions.CategoricalDistribution([True, False]),
    "random_forest__max_depth": optuna.distributions.IntDistribution(10, 100, 10),
    # 'random_forest__max_features': optuna.distributions.CategoricalDistribution(['auto', 'sqrt']),
    "random_forest__min_samples_leaf": optuna.distributions.IntDistribution(2, 10),
    "random_forest__min_samples_split": optuna.distributions.IntDistribution(2, 10),
    "random_forest__n_estimators": optuna.distributions.IntDistribution(10, 100, 10),
}

params_RF_BS = {
    # 'random_forest__bootstrap': Categorical([True, False]),
    "random_forest__max_depth": Integer(10, 100),
    # 'random_forest__max_features': Categorical(['auto', 'sqrt']),
    "random_forest__min_samples_leaf": Integer(2, 10),
    "random_forest__min_samples_split": Integer(2, 10),
    "random_forest__n_estimators": Integer(10, 100),
}

params_RF_GS = {
    # 'random_forest__bootstrap': [True],
    "random_forest__max_depth": [10, 100],
    # 'random_forest__max_features': ['auto', 'sqrt'],
    "random_forest__min_samples_leaf": [2, 10],
    "random_forest__min_samples_split": [2, 10],
    "random_forest__n_estimators": [10, 100],
}


PARAMS = {
    ("ligthgbm", "Optuna"): params_LGM_OPT,
    ("random_forest", "Optuna"): params_RF_OPT,
    ("xgboost", "Optuna"): params_XGB_OPT,
    ("ligthgbm", "BayesSearch"): params_LGM_BS,
    ("random_forest", "BayesSearch"): params_RF_BS,
    ("ligthgbm", "GridSearch"): params_LGM_GS,
    ("random_forest", "GridSearch"): params_RF_GS,
}

# ('ligthgbm-binary', 'Optuna'): params_LGM_OPT ,('random_forest-binary', 'Optuna'): params_RF_OPT,
#     ('ligthgbm-binary', 'BayesSearch'): params_LGM_BS ,('random_forest-binary', 'BayesSearch'): params_RF_BS,
#     ('ligthgbm-binary', 'GridSearch'): params_LGM_GS ,('random_forest-binary', 'GridSearch'): params_RF_GS,
#     ('ligthgbm-multiclass', 'Optuna'): params_LGM_OPT ,('random_forest-multiclass', 'Optuna'): params_RF_OPT,
#     ('ligthgbm-multiclass', 'BayesSearch'): params_LGM_BS ,('random_forest-multiclass', 'BayesSearch'): params_RF_BS,
#     ('ligthgbm-multiclass', 'GridSearch'): params_LGM_GS ,('random_forest-multiclass', 'GridSearch'): params_RF_GS,
#     ('ligthgbm-regression', 'Optuna'): params_LGM_OPT ,('random_forest-regression', 'Optuna'): params_RF_OPT,
#     ('ligthgbm-regression', 'BayesSearch'): params_LGM_BS ,('random_forest-regression', 'BayesSearch'): params_RF_BS,
#     ('ligthgbm-regression', 'GridSearch'): params_LGM_GS ,('random_forest-regression', 'GridSearch'): params_RF_GS}


def get_model_params(regressor, search):

    params = PARAMS[(regressor, search)]

    return params
