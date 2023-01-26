"""
TODO: Add docstrings (JAVIER)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from category_encoders import BinaryEncoder, GLMMEncoder, OrdinalEncoder, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (  # SelectorMixin,
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder  # categorical features conversion
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler  # scale,

from ceramic import logger
from ceramic.ml_prediction.training.common_model import get_predictor

# Categorical Encoding Methods
# from category_encoders import (OrdinalEncoder)
#
# from collections import defaultdict
# from typing import Dict, List, Optional


log = logger.get_logger(__name__)


class TargetTransformer:
    """
    Perform some transformation on the time series
    data in order to make the model more performant and
    avoid non-stationary effects.
    """

    def __init__(self, log=False, detrend=False, deseason=False, diff=False):

        self.trf_log = log
        self.trf_detrend = detrend
        self.deseason = deseason
        self.trend = pd.Series(dtype=np.float64)

    def transform(self, index, values):
        """
        Perform log transformation to the target time series

        index: the index for the resulting series
        values: the values of the initial series

        Return:
            transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_detrend:
            self.trend = TargetTransformer.get_trend(res) - np.mean(res.values)
            res = res.subtract(self.trend)

        if self.trf_log:
            res = pd.Series(index=index, data=np.log(res.values))

        return res

    def inverse(self, index, values):
        """
        Go back to the original time series values

        :param index: the index for the resulting series
        :param values: the values of series to be transformed back

        Return:
            inverse transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_log:
            res = pd.Series(index=index, data=np.exp(values))
        try:
            if self.trf_detrend:
                assert len(res.index) == len(self.trend.index)
                res = res + self.trend

        except AssertionError:
            print("Use a different transformer for each target to transform")

        return res

    @staticmethod
    def get_trend(data):
        """
        Get the linear trend on the data which makes the time
        series not stationary
        """
        n = len(data.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.Series(index=data.index, data=trend)


def remove_decimals(df):

    df = df.copy()

    df = df.round(0)
    df = df.dropna()

    return df


class Scaler_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type="min-max"):

        super().__init__()

        self.scaler_type = scaler_type

        if scaler_type == "min-max":
            self._scaler_model = MinMaxScaler()  # .set_output(transform='pandas')
        elif scaler_type == "standard":
            self._scaler_model = StandardScaler()  # .set_output(transform='pandas')
        elif scaler_type == "power":
            self._scaler_model = PowerTransformer()  # .set_output(transform='pandas')
        elif scaler_type == "quantile_normal":
            self._scaler_model = QuantileTransformer(
                output_distribution="normal"
            )  # .set_output(transform='pandas') #same as PowerTransformer()
        elif scaler_type == "quantile_uniform":
            self._scaler_model = QuantileTransformer(output_distribution="uniform")  # .set_output(transform='pandas')

    def fit(self, X, y=None):

        if self.scaler_type in ["quantile_normal", "quantile_uniform"]:
            self._scaler_model.set_params(n_quantiles=X.shape[0] // 2)
        self._scaler_model.fit(X)

        return self

    def transform(self, X):
        """
        Normalize features
        """
        data = X.copy()

        data_norm = self._scaler_model.transform(data)

        # data[:] = data_norm

        return data_norm

    def inverse_transform(self, X):

        data_norm = self._scaler_model.inverse_transform(X)

        return data_norm


class Encoding_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_type="one_hot", categories: str | list = "auto", mapping: list = [], **params):

        super().__init__()

        self.encoder_type = encoder_type
        self.categories = categories
        self.mapping = mapping

        if self.encoder_type == "one_hot":
            self._encoder_model = OneHotEncoder(
                categories="auto", sparse_output=False, drop="first", handle_unknown="ignore"
            )  # .set_output(transform='pandas')
        elif self.encoder_type == "ordinal":  # Better than one hot for high cardinality
            self._encoder_model = OrdinalEncoder(mapping, handle_unknown="ignore")
        elif self.encoder_type == "binary":  # Better than one hot for high cardinality
            self._encoder_model = BinaryEncoder()
        elif self.encoder_type == "GLM":
            self._encoder_model = GLMMEncoder()
        elif self.encoder_type == "target":
            self._encoder_model = TargetEncoder()

    def fit(self, X, y=None):

        if self.encoder_type == "target":
            self._encoder_model.fit(X, y)
        else:
            self._encoder_model.fit(X)

        return self

    def transform(self, X):
        """
        Encode features
        """
        data = X.copy()

        data_norm = self._encoder_model.transform(data)

        return data_norm

    def inverse_transform(self, X):

        data_norm = self._encoder_model.inverse_transform(X)

        return data_norm


class Features_Creation_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, method: str = "MI", normalized=False):
        super().__init__()
        self.target = target
        self.method = method
        self.normalized = normalized

    def get_features(self, data):

        """
        Create the feature set for data

        Parameters
        ----------
        data: pd.Dataframe

        Returns
        -------
        features: pd.DataFrame with the feature set
        target: pd.Series holding the target time series with the same index as the features
        """

        # TODO Include conversion of categorical variables

        df_data = data.copy()

        features = pd.DataFrame()

        ts_features = self.create_ts_features(df_data)
        features = features.join(ts_features, how="outer")

        features.dropna(inplace=True)

        target = df_data.loc[features.index, self.target]

        return features, target

    def create_ts_features(self, data):

        ts_features = pd.DataFrame()

        if self.normalized is False:

            ts_features["hour"] = data.index.hour
            ts_features["weekday"] = data.index.weekday
            # ts_features["dayofyear"] = data.index.dayofyear
            ts_features["is_weekend"] = data.index.weekday.isin([5, 6]).astype("int32")  # changed from np.int32
            ts_features["weekofyear"] = data.index.weekofyear
            ts_features["month"] = data.index.month
            ts_features["season"] = (data.index.month % 12 + 3) // 3

        else:

            ts_features["hour_cos"] = np.round(np.cos(data.index.hour * (2 * np.pi / 24)), 3)
            ts_features["hour_sin"] = np.round(np.sin(data.index.hour * (2 * np.pi / 24)), 3)
            ts_features["is_weekend"] = data.index.weekday.isin([5, 6]).astype("int32")  # changed from np.int32
            ts_features["weekofyear_cos"] = np.round(np.cos(data.index.weekofyear * (2 * np.pi / 52)), 3)
            ts_features["weekofyear_sin"] = np.round(np.sin(data.index.weekofyear * (2 * np.pi / 52)), 3)
            ts_features["monthofyear_cos"] = np.round(np.cos(data.index.month * (2 * np.pi / 12)), 3)
            ts_features["monthofyear_sin"] = np.round(np.sin(data.index.month * (2 * np.pi / 12)), 3)
            day_names = ["d1", "d2", "d3", "d4", "d5", "d6"]
            features_weekday = pd.DataFrame(
                self.weekday_encoder.transform(np.array(data.index.weekday).reshape(-1, 1))
            ).set_axis(day_names, axis=1)
            ts_features = ts_features.join(features_weekday, how="outer")
            season_names = ["Sea1", "Sea2", "Sea3"]
            features_season = pd.DataFrame(
                self.season_encoder.transform(np.array((data.index.month % 12 + 3) // 3).reshape(-1, 1))
            ).set_axis(season_names, axis=1)
            ts_features = ts_features.join(features_season, how="outer")

        ts_features.index = data.index

        return ts_features

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        features, target = self.get_features(X)

        return features, target


# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/


class Features_Selection_Transformer(BaseEstimator, TransformerMixin):
    """Class for feature selection

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """

    def __init__(self, method="RFE", base_predictor: str = "ligthgbm", objective: str = "regression", **params):

        super().__init__()
        self.method = method
        self.base_predictor = base_predictor
        self.objective = objective
        self.model = get_predictor(base_predictor, objective, params)

    def __sklearn_is_fitted__(
        self,
    ):  # to avoid NotFitted error from check_is_fitted() in FeatureImportance class in ml_explainer
        return True

    def get_support(self):

        mask = self.feat.get_support(indices=False)

        return mask

    def fit(self, X, y=None):

        self.num_features = int(X.shape[1] * 3 / 4)

        if isinstance(X, pd.DataFrame) and self.base_predictor == "ligthgbm":
            X = X.to_numpy()
        # Wrapper methods
        if self.method == "RFE":
            self.feat = RFE(self.model, n_features_to_select=self.num_features).fit(X, y)
        elif self.method == "RFECV":
            self.feat = RFECV(self.model, n_jobs=-1).fit(X, y)
        elif self.method == "SHAP":
            # DF, based on which importance is checked
            self.model.fit(X, y)
        elif self.method == "PFI":  # Permutation Feature Importance
            self.feat = permutation_importance(self.model, X, y)
            # TODO #ELI5
        elif self.method == "SFM":
            self.feat = SelectFromModel(estimator=self.model).fit(X, y)
        elif self.method == "SFS":  # Sequential Feature Selection
            pass
            # TODO

        # Filter statistical methods
        elif self.method == "chi2":  # Chi-Squared (categorical-classification)
            self.feat = SelectKBest(score_func=chi2, k=self.num_features).fit(X, y)
        elif self.method == "MI":  # Mutual information (categorical-classification)
            self.feat = SelectKBest(score_func=mutual_info_classif, k=self.num_features).fit(X, y)
        elif self.method == "ANOVA":  # numerical-classification/regression)
            self.feat = SelectKBest(score_func=f_classif, k=self.num_features).fit(X, y)
        elif self.method == "SP":  # Selection Percentile (categorical-classification)
            self.feat = SelectPercentile(chi2, percentile=self.num_features).fit(X, y)
        elif self.method == "VT":  # Variance Threshold
            self.feat = VarianceThreshold(threshold=0.1).fit(X, y)
        elif self.method == "DCI":  # Drop Column Importance
            pass
            # TODO

        else:
            self.model.fit(X, y)

        return self

    def transform(self, X):

        if self.method == "SHAP":

            if isinstance(X, pd.DataFrame):  # and (self.base_predictor == 'ligthgbm'):
                X = X.to_numpy()

            # Explain model predictions using shap library:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)  # (X_test)
            shap_sum = np.abs(shap_values).mean(axis=0)
            """
            importance_df = pd.DataFrame(shap_sum.tolist(), index = X.columns.tolist(), columns = ['SHAP_importance'])
            importance_df = importance_df.sort_values('SHAP_importance', ascending=False)
            importance = importance_df[importance_df['SHAP_importance']>0].index
            features = X[importance]
            """
            self.indices = np.argsort(shap_sum)[::-1][: self.num_features]
            features = X[:, np.argsort(shap_sum)[::-1][: self.num_features]]

        elif self.method in ["RFE", "RFECV", "SP", "VT", "chi2", "MI", "ANOVA", "SFM"]:

            if (isinstance(X, pd.DataFrame)) and (self.base_predictor == "ligthgbm"):
                X = X.to_numpy()

            features = self.feat.transform(X)
            self.indices = self.feat.get_support(indices=True)

            if isinstance(features, pd.DataFrame) and self.base_predictor == "ligthgbm":
                features = features.to_numpy()

        elif self.method == "PFI":
            pass

        else:

            importance_df = pd.DataFrame([X.columns.tolist(), self.model.feature_importances_]).T
            importance_df.columns = ["column_name", "feature_importance"]
            self.importance = importance_df.sort_values("feature_importance", ascending=False)

        # log.info(f'FEATURES: {features.columns}')

        return features

    """
    def inverse(self, data_norm, col_names):

        data_dummy = pd.DataFrame(np.zeros((len(data_norm), len(col_names))), columns=col_names)
        data_dummy[self.target_name] = data_norm
        data_denorm = pd.DataFrame(self.min_max_data.inverse_transform(data_dummy), columns=col_names)

        return data_denorm[self.target_name].values
    """
