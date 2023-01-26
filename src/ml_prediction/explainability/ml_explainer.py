"""
MODULE EXPLAINER. Regression
Javier Monreal Tolmo, Angel Martinez-Tenor
TODO: Add docstrings (ANGEL / JAVIER)
"""

from __future__ import annotations

import dalex as dx
import numpy as np
import pandas as pd

# import plotly.express as px  # The charts will be needed in the UI only
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ceramic import logger

shap.initjs()


log = logger.get_logger(__name__)

# ---------------- SHAP ---------------------#


def generate_explainer(pipeline: Pipeline, X):
    """Generate an explainer based on Shapley Values

    Args:
        pipeline (sklearn.pipeline.Pipeline): scikit pipeline
        X (pd.DataFrame): training dataset

    Returns:
        shap.TreeExplainer: shap explainer
    """

    print("\nGenerating Explainer with all samples (Shapley)")

    x_transformed = pipeline[:-1].transform(X)

    feature_importance = FeatureImportance(pipeline, verbose=False)
    transformed_features = feature_importance.get_selected_features()

    explainer = shap.TreeExplainer(
        pipeline[-1],
        data=x_transformed,
        feature_names=transformed_features,
        feature_perturbation="interventional",
        # model_output="probability",
    )
    return explainer


def generate_summary_shap_plot(
    pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_features: str | list = "fritname"
):
    """Generate plots for global contribution of the Shapley Values

    Args:
        pipeline (sklearn.pipeline.Pipeline): scikit pipeline
        X_train (pd.DataFrame): training dataset
        X_test (pd.DataFrame): test dataset
        categorical_features (str, optional): categorical feature/s processed by OHE. Defaults to 'fritname'.
    """

    explainer = generate_explainer(pipeline, X_train)

    # x_train_transformed = pipeline[:-1].transform(X_train)
    x_test_transformed = pipeline[:-1].transform(X_test)

    shap_values = explainer(x_test_transformed, check_additivity=False)

    if categorical_features is not None:
        shap_values, sv_ori = combine_one_hot(
            shap_values, categorical_features, [categorical_features in n for n in shap_values.feature_names]
        )

    if shap_values.values.ndim == 3:  # differences between rf & lightgbm
        shap_values = shap_values[:, :, 1]

    shap.plots.beeswarm(shap_values)
    shap.summary_plot(shap_values)
    shap.plots.bar(shap_values)
    # bar_shap_abs(shap_values, X_test)


def shap_plot_single_predictions(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    j: int,
    type: str = "force",
    categorical_features: str | list = "fritname",
):
    """Generate plot for single prediction contribution based on Shapley Values

    Args:
        pipeline (sklearn.pipeline.Pipeline): scikit pipeline
        X_train (pd.DataFrame): training dataset
        X_test (pd.DataFrame): test dataset
        j (int): prediction index
        type (str, optional): type of shap plot for single prediction. Defaults to 'force'.
        categorical_features (str, optional): categorical feature/s processed by OHE. Defaults to 'fritname'.

    Returns:
        shap plot
    """

    explainer = generate_explainer(pipeline, X_train)

    # x_train_transformed = pipeline[:-1].transform(X_train)
    x_test_transformed = pipeline[:-1].transform(X_test)

    shap_values = explainer(x_test_transformed)

    if type == "bar":
        shap_values, sv_ori = combine_one_hot(
            shap_values, categorical_features, [categorical_features in n for n in shap_values.feature_names]
        )
        p = shap.plots.bar(shap_values[j])
    elif type == "force":
        # shap_values = explainer.shap_values(x_test_transformed)
        # shap_values, sv_ori = combine_one_hot(shap_values, categorical_features, [categorical_features in n for n in
        # shap_values.feature_names])
        p = shap.force_plot(
            explainer.expected_value, shap_values.values[j, :], x_test_transformed.iloc[j, :]
        )  # x_test_transformed es pd.DataFrame --> set_config(transform_output="pandas")
    # if type == 'water':
    #     shap_values, sv_ori = combine_one_hot(shap_values, categorical_features, [categorical_features in n for n in
    # shap_values.feature_names])
    #     p = shap.waterfall_plot(shap_values[j])
    # elif type == 'force_all':
    #     #shap_values = explainer.shap_values(x_test_transformed)
    #     #shap_values, sv_ori = combine_one_hot(shap_values, categorical_features, [categorical_features in n for n in
    # shap_values.feature_names])
    #     p = shap.force_plot(explainer.expected_value, shap_values.values[:,:], x_test_transformed[:,:].iloc[j,:])

    return p


# '''
# def save_explainer(explainer, filename=PATH_EXPLAINER, shap_values=False):
#     """Save the explainer"""

#     joblib.dump(explainer, filename, compress=True)
#     print(f"Explainer Saved: \t{filename}")


# def load_explainer(filename=PATH_EXPLAINER):
#     """Load the explainer"""

#     explainer = joblib.load(filename)
#     print(f"Explainer Loaded:\t{filename}")
#     return explainer

# '''


def bar_shap_abs(df_shap, df_X: pd.DataFrame):
    """Show bar shap plot with color depending oin positive or negative contribution

    Args:
        df_shap: Shapley values
        df_X (pd.DataFrame): original test dataset
    """
    # import matplotlib as plt
    # Make a copy of the input data
    df = df_X.dropna(axis=1)
    shap_v = pd.DataFrame(df_shap.values)
    feature_list = df.columns
    shap_v.columns = feature_list

    df_v = df.copy().reset_index().drop("time", axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i, feat in enumerate(feature_list):
        # print(f'DF: {df_v.shape}')
        # print(f'SHAP: {shap_v.columns}')

        a = np.corrcoef(shap_v[feat], df_v[feat])
        b = a[1][0]
        # b = np.corrcoef(shap_v[i,:],df_v.iloc[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["Variable", "Corr"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, "red", "blue")

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["Variable", "SHAP_abs"]
    k2 = k.merge(corr_df, left_on="Variable", right_on="Variable", how="inner")
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    colorlist = k2["Sign"]
    ax = k2.plot.barh(x="Variable", y="SHAP_abs", color=colorlist, figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")


class FeatureImportance:

    """

    Extract & Plot the Feature Names & Importance Values from a Scikit-Learn Pipeline.

    The input is a Pipeline that starts with a ColumnTransformer & ends with a regression or classification model.
    As intermediate steps, the Pipeline can have any number or no instances from sklearn.feature_selection.

    Note:
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely
    new columns, it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator &
    SimpleImputer(add_indicator=True) add columns to the dataset that didn't exist before, so there should come last in
    the Pipeline.


    Parameters
    ----------
    pipeline : a Scikit-learn Pipeline class where the a ColumnTransformer is the first element and model estimator is
    the last element verbose : a boolean. Whether to print all of the diagnostics. Default is False.

    Attributes
    __________
    column_transformer_features :  A list of the feature names created by the ColumnTransformer prior to any selectors
    being applied
    transformer_list : A list of the transformer names that correspond with the
    `column_transformer_features` attribute
    discarded_features : A list of the features names that were not selected by a sklearn.feature_selection instance.
    discarding_selectors : A list of the selector names corresponding with the `discarded_features` attribute
    feature_importance :  A Pandas Series containing the feature importance values and feature names as the index.
    plot_importances_df : A Pandas DataFrame containing the subset of features and values that are actually displaced
    in the plot.
    feature_info_df : A Pandas DataFrame that aggregates the other attributes. The index is column_transformer_features.
     The transformer column contains the transformer_list. value contains the feature_importance values.
     discarding_selector contains discarding_selectors & is_retained is a Boolean indicating whether the feature was
     retained.

    """

    def __init__(self, pipeline, verbose=False):
        self.pipeline = pipeline
        self.verbose = verbose

    def get_feature_names(self, verbose=None):

        """
        Get the column names from the a ColumnTransformer containing transformers & pipelines

        Parameters
        ----------
        verbose : a boolean indicating whether to print summaries.
            default = False


        Returns
        -------
        a list of the correct feature names

        Note:
        If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely
        new columns, it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator &
        SimpleImputer(add_indicator=True) add columns to the dataset that didn't exist before, so there should come
        last in the Pipeline.
        Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525
        """

        if verbose is None:
            verbose = self.verbose

        if verbose:
            print("""\n\n---------\nRunning get_feature_names\n---------\n""")

        column_transformer = self.pipeline[0]
        assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
        check_is_fitted(column_transformer)

        new_feature_names, transformer_list = [], []

        for i, transformer_item in enumerate(column_transformer.transformers_):

            transformer_name, transformer, orig_feature_names = transformer_item
            orig_feature_names = list(orig_feature_names)

            if verbose:
                print(
                    "\n\n", i, ". Transformer/Pipeline: ", transformer_name, ",", transformer.__class__.__name__, "\n"
                )
                print("\tn_orig_feature_names:", len(orig_feature_names))

            if transformer == "drop":

                continue

            if isinstance(transformer, Pipeline):
                # if pipeline, get the last transformer in the Pipeline
                transformer = transformer.steps[-1][1]

            if hasattr(transformer, "get_feature_names_out"):
                if "input_features" in transformer.get_feature_names_out.__code__.co_varnames:
                    names = list(transformer.get_feature_names_out(orig_feature_names))
                else:
                    names = list(transformer.get_feature_names_out())

            if hasattr(transformer, "get_feature_names"):

                if "input_features" in transformer.get_feature_names.__code__.co_varnames:

                    names = list(transformer.get_feature_names(orig_feature_names))

                else:

                    names = list(transformer.get_feature_names())

            elif hasattr(transformer, "indicator_") and transformer.add_indicator:
                # is this transformer one of the imputers & did it call the MissingIndicator?

                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [orig_feature_names[idx] + "_missing_flag" for idx in missing_indicator_indices]
                names = orig_feature_names + missing_indicators

            elif hasattr(transformer, "features_"):
                # is this a MissingIndicator class?
                missing_indicator_indices = transformer.features_
                missing_indicators = [orig_feature_names[idx] + "_missing_flag" for idx in missing_indicator_indices]

                print(f"MISSING: {missing_indicators}")

            else:

                names = orig_feature_names

            if verbose:
                print("\tn_new_features:", len(names))
                print("\tnew_features:\n", names)

            new_feature_names.extend(names)
            transformer_list.extend([transformer_name] * len(names))

        self.transformer_list, self.column_transformer_features = transformer_list, new_feature_names

        return new_feature_names

    def get_selected_features(self, verbose=None):
        """

        Get the Feature Names that were retained after Feature Selection (sklearn.feature_selection)

        Parameters
        ----------
        verbose : a boolean indicating whether to print summaries. default = False

        Returns
        -------
        a list of the selected feature names


        """

        if verbose is None:
            verbose = self.verbose

        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        features = self.get_feature_names()

        if verbose:
            print("\n\n---------\nRunning get_selected_features\n---------\n")

        all_discarded_features, discarding_selectors = [], []

        for i, step_item in enumerate(self.pipeline.steps[:]):

            step_name, step = step_item

            if hasattr(step, "get_support"):

                if verbose:
                    print("\nStep ", i, ": ", step_name, ",", step.__class__.__name__, "\n")

                check_is_fitted(step)

                feature_mask_dict = dict(zip(features, step.get_support()))

                features = [feature for feature, is_retained in feature_mask_dict.items() if is_retained]

                discarded_features = [feature for feature, is_retained in feature_mask_dict.items() if not is_retained]

                all_discarded_features.extend(discarded_features)
                discarding_selectors.extend([step_name] * len(discarded_features))

                if verbose:
                    print(f"\t{len(features)} retained, {len(discarded_features)} discarded")
                    if len(discarded_features) > 0:
                        print("\n\tdiscarded_features:\n\n", discarded_features)

        self.discarded_features, self.discarding_selectors = all_discarded_features, discarding_selectors

        return features


def combine_one_hot(shap_values, name: str, mask: list, return_original: bool = True):
    """This function assumes that shap_values[:, mask] make up a one-hot-encoded feature

    Args:
        shap_values: Shapley values
        name (str): categorical feature/s processed by OHE
        mask (list): bool array same length as features

    Returns:
        new Shapley values with OHE variables for categorical feature combined in unique variable
    """

    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]

    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_col_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
        instance_names=shap_values.instance_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )

    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)

    svdata = np.concatenate([shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1)

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate([svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1)

    new_values = sv_name.values.sum(axis=1)
    svvalues = np.concatenate([shap_values.values[:, ~mask], new_values.reshape(-1, 1)], axis=1)
    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(
        svvalues,
        base_values=shap_values.base_values,
        data=svdata,
        display_data=svdisplay_data,
        instance_names=shap_values.instance_names,
        feature_names=svfeature_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    if return_original:
        return sv, sv_name
    else:
        return sv


# ---------------- DALEX ---------------------#


def dalex_explainer(pipeline, X, y):

    explainer = dx.Explainer(pipeline, X, y)

    return explainer


def variable_importance(explainer, variables: None | list = None, type: str = "variable_importance"):

    if type == "shap_wrapper":
        log.exception("Shapp values not implemented for Pipelines")

    vi = explainer.model_parts(variables=variables, type=type)
    vi.result

    vi.plot()


def model_profile(explainer, type: str, label: str):

    profile = explainer.model_profile(type=type, label=label)

    return profile
