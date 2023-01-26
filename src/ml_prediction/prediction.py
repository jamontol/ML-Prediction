import gzip
import os
import pickle

# import re
from pathlib import Path

import joblib

# import numpy as np
# import pandas as pd
from collinearity import SelectNonCollinear
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn import set_config
from sklearn.base import BaseEstimator, RegressorMixin  # ClassifierMixin,
from sklearn.compose import ColumnTransformer, make_column_selector  # TransformedTargetRegressor,
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer # target conversion
# (only if needed)
# from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from ceramic import logger

from .featuring.features import (  # TargetTransformer,
    Encoding_Transformer,
    Features_Creation_Transformer,
    Features_Selection_Transformer,
    Scaler_Transformer,
)
from .training.common_model import get_predictor, training_model

# from typing import Dict, List, Optional


# from feature_engine.selection import (DropFeatures, DropConstantFeatures, DropDuplicateFeatures)


# from mlxtend.classifier import EnsembleVoteClassifier


log = logger.get_logger(__name__)


class ML_predictor(BaseEstimator, RegressorMixin):
    """Class for ML regression

    Args:
        target (str | list): target/s variable to be predicted
        objective (str): type of prediction objective. Defaults to 'regression'.
        base_predictor (str): base ML predictor. Defaults to 'ligthgbm'.
        variables_type (dict): dict with type of variables. Defaults to {}.
        select_features (None | str | dict): features selection method . Defaults to ['RFE','ANOVA'].
        features_creation (None | bool): whether to carry out features creation (True) or not (False). Defaults to False
        scaler (str | None): scaler method. Defaults to 'min-max'.
        collinear (bool): whether to select non collinear features(True) or not (False). Defaults to False.
        sampling (bool): whether to do sampling for data imbalance (True) or not (False), only for classification
        objectives. Defaults to False.
        interval_strategy (dict | None): MAPIE interval strategy. Defaults to None.
        HPO (str | None): method for HPO tuning. Defaults to None.
        params (dict): params for base predictor. Defaults to {}.

    """

    def __init__(
        self,
        target: str | list,
        objective: str = "binary",
        base_predictor: str = "ligthgbm",
        variables_type: dict = {},
        select_features: None | str | dict = ["RFE", "ANOVA"],
        features_creation: None | bool = False,
        scaler: str | None = "min-max",
        encoder: str = "one_hot",
        collinear: bool = False,
        sampling: bool = False,
        interval_strategy: dict | None = None,
        HPO: str | None = None,
        params: dict = {},
    ):

        self.featuring = []
        self.base_predictor = base_predictor

        self.objective = objective
        self.target = target

        self.remove_columns = []
        self.variables_type = variables_type
        self.categorical_variables = list(variables_type["category"])
        self.numeric_variables = list(variables_type["numeric"])
        self.ordinal_variables = []
        if variables_type["ordinal"] != []:
            self.ordinal_variables = [dict["col"] for dict in variables_type["ordinal"]]
        self.ordinal_mapping = variables_type["ordinal"]

        self.features_creation = features_creation
        self.select_features = select_features
        self.scaler = scaler
        self.encoder = encoder
        self.collinear = collinear
        self.sampling = sampling
        self.HPO = HPO
        self.interval_strategy = interval_strategy

        # mlflow.log_param("ML_params")#, self.__dict__)

        self.build_pipeline(params)

    def build_pipeline(self, params: dict):
        """Build scikit pipeline

        Args:
            params (dict): params for base predictor model
        """
        set_config(transform_output="pandas")  # DataFrame output for transformers for SHAP
        pipeline_steps = list()

        # pipeline_steps.append(('drop_features', make_pipeline(DropFeatures([self.remove_columns]),
        # DropConstantFeatures(tol=1, missing_values='ignore'), DropDuplicateFeatures())))

        if self.scaler is not None:

            column_scaler = ColumnTransformer(
                [
                    (
                        "numerical",
                        make_pipeline(SimpleImputer(strategy="mean"), Scaler_Transformer(scaler_type=self.scaler)),
                        self.numeric_variables,
                    ),
                    (
                        "categorical",
                        make_pipeline(
                            SimpleImputer(strategy="most_frequent"), Encoding_Transformer(encoder_type=self.encoder)
                        ),
                        self.categorical_variables,
                    ),
                    (
                        "ordinal",
                        make_pipeline(
                            SimpleImputer(strategy="most_frequent"),
                            Encoding_Transformer(encoder_type="ordinal", mapping=self.ordinal_mapping),
                        ),
                        self.ordinal_variables,
                    ),
                ],
                verbose_feature_names_out=False,
            )
        else:
            column_scaler = ColumnTransformer(
                [
                    (
                        "categorical",
                        make_pipeline(
                            SimpleImputer(strategy="most_frequent"), Encoding_Transformer(encoder_type=self.encoder)
                        ),
                        self.categorical_variables,
                    ),
                    (
                        "ordinal",
                        make_pipeline(
                            SimpleImputer(strategy="most_frequent"),
                            Encoding_Transformer(encoder_type="ordinal", mapping=self.ordinal_mapping),
                        ),
                        self.ordinal_variables,
                    ),
                ]
            )

        pipeline_steps.append(("scaler", column_scaler))

        if self.select_features is not None:

            if isinstance(self.select_features, str):  # Same features selection for all features

                pipeline_steps.append(
                    (
                        "select_features",
                        Features_Selection_Transformer(
                            base_predictor=self.base_predictor, method=self.select_features, **params
                        ),
                    )
                )

            else:

                column_selector = ColumnTransformer(
                    [
                        (
                            "categorical_features",
                            Features_Selection_Transformer(
                                base_predictor=self.base_predictor, method=self.select_features["categorical"], **params
                            ),
                            make_column_selector(dtype_include="category"),
                        )
                    ],  # self.categorical_variables+self.ordinal_variables)], #categoricals no converted to one-hot
                    remainder=Features_Selection_Transformer(
                        base_predictor=self.base_predictor, method=self.select_features["numerical"], **params
                    ),
                )  # including those categorical converted to one-hot
                pipeline_steps.append(("select_features", column_selector))

            # pipeline_steps.append(('select_features', Features_Selection_Transformer(
            # base_regressor=self.base_regressor, method=self.select_features, **params)))

        if self.collinear:  # after featuring

            pipeline_steps.append(("collinearity", SelectNonCollinear(correlation_threshold=0.4, scoring=f_regression)))

        if self.sampling:

            pipeline_steps.append(("sampling", SMOTE()))

        pipeline_steps.append((f"{self.base_predictor}", get_predictor(self.base_predictor, self.objective, params)))

        if self.objective in ["binary", "multiclass", "multilabel"]:
            if self.sampling:
                self.pipeline = imbPipeline(steps=pipeline_steps)  # To include SMOTE sampling for data imbalance
            else:
                self.pipeline = Pipeline(steps=pipeline_steps)
        elif self.objective == "regression":
            if self.scaler is not None:
                self.pipeline = Pipeline(steps=pipeline_steps)
                # self.pipeline = TransformedTargetRegressor(regressor=self.pipeline,
                # transformer=Scaler_Transformer(scaler_type=self.scaler))
            else:
                self.pipeline = Pipeline(steps=pipeline_steps)

        # pipeline.named_steps
        # pipeline.steps[0][0]

    @classmethod
    def load(cls, model_path: str, model: str, type: str = "pickle"):

        log.info(f"LOADING MODEL {model}")
        log.info(f"PATH {model_path}")

        # name = os.path.basename(model_path)

        if os.path.exists(model_path):
            if type == "pickle":
                serial_file = Path(model_path) / f"ML_model_{model}.pkl"
                with gzip.open(serial_file, "rb") as f:
                    forecaster_model = pickle.load(f)
            elif type == "joblib":
                serial_file = Path(model_path) / f"ML_model_{model}.joblib"
                with gzip.open(serial_file, "rb") as f:
                    forecaster_model = joblib.load(f)

            print("ML model successfully loaded")

        else:
            error = "Path doesn't exist"
            print(error)
            return error

        return forecaster_model

    def save(self, save_dir: str, model: str, type: str = "pickle"):

        print("Saving ML Model ...", sep=" ")
        os.makedirs(save_dir, exist_ok=True)

        if type == "pickle":
            model_name = f"ML_model_{model}.pkl"
            file_path = Path(save_dir) / model_name
            with gzip.open(file_path, "wb") as f:
                # Pickle the ML model using the highest protocol available.
                pickle.dump(self, f)  # TODO dill
        elif type == "joblib":
            model_name = f"ML_model_{model}.joblib"
            file_path = Path(save_dir) / model_name
            with gzip.open(file_path, "wb") as f:
                # Pickle the ML model using the highest protocol available.
                joblib.dump(self, f, compress=0)  # TODO dill

        print(f"ML model successfully saved as {model_name}")

    def fit(self, X_train, y_train, params: dict, scoring: str = None, shap: bool = False):

        self.params = params

        # mlflow.log_param("ML_regressor_params", self.params)

        if self.features_creation:
            self.featuring = Features_Creation_Transformer(target=self.target, normalized=self.scaler)
            X_train, _ = self.featuring.fit_transform(X_train)

        if self.objective == "binary":
            self.model, eval_score, best_params = training_model(
                self.pipeline, X_train, y_train, objective=self.objective, scoring=scoring, HPO=self.HPO
            )

        elif self.objective == "multiclass":
            if self.base_predictor == "ligthgbm":
                assert params["objective"] == "multiclass", log.critical("The objective param must be 'multiclass'")
            self.model, eval_score, best_params = training_model(self.pipeline, X_train, y_train, HPO=self.HPO)

        elif self.objective == "multilabel":
            self.model, eval_score, best_params = training_model(
                MultiOutputClassifier(self.pipeline),
                X_train,
                y_train,
                objective=self.objective,
                scoring=scoring,
                HPO=self.HPO,
            )

        elif self.objective == "regression":
            self.model, eval_score, best_params = training_model(
                self.pipeline,
                X_train,
                y_train,
                objective=self.objective,
                scoring=scoring,
                HPO=self.HPO,
                interval_strategy=self.interval_strategy,
            )

        log.info(f"SCORE: {eval_score:.2f}")
        # log.info(f'PARAMS: {best_params}')
        # if shap:
        #     self.rf_predict_test_set(X, y, compute_shap=True)

    # --------------------------- PREDICTION ---------------------------------

    def predict(self, data, prob=False, df_pred=None):

        if prob:
            predictions = self.model.predict_proba(df_pred)[:, 1]
        else:
            predictions = self.model.predict(data)

        return predictions
