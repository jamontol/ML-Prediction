"""
TODO: Add docstrings (JAVIER)
"""

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

# import os
# import pickle
from math import sqrt

import numpy as np
import pandas as pd

# from mlxtend.data import autompg_data
from mlxtend.evaluate import bootstrap, paired_ttest_5x2cv
from sklearn.metrics import (  # roc_auc_score,
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ceramic import logger

from .binaryclass import evaluator as binary_eval

# from .check import label_check
from .multilabel import evaluator as multiclass_eval
from .multilabel import evaluator as multilabel_eval

log = logger.get_logger(__name__)

# mlxtend


def confidence_interval(metric: float, n: int, z: float = 1.96, type: str = "parametric", X=None, eval_func=None):
    """CI of classification/regression metric

    Args:
        metric (float): value of metric
        n (int): size of the sample
        z (float, optional): number of standard deviations from the Gaussian distribution. Defaults to 1.96.
    """
    if type == "parametric":

        interval = z * sqrt((metric * (1 - metric)) / n)
        print("%.3f" % interval)

    else:  # bootstrap

        original, std_err, ci_bounds = bootstrap(X, num_rounds=1000, func=eval_func, ci=0.95, seed=123)

        print(f"Mean: {original:.2f}, SE: +/- {std_err:.2f}, CI95: [{ci_bounds[0]:.2f}, {ci_bounds[1]:.2f}]")


def paired_t_test(clf1, clf2, X, y):
    """5x2cv paired t test procedure to compare the performance of two models

    Args:
        clf1 (_type_): model 1
        clf2 (_type_): model 2
        X (_type_): feature data
        y (_type_): target data
    """

    t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf2, X=X, y=y, random_seed=1)

    print("t statistic: %.3f" % t)
    print("p value: %.3f" % p)


# scikit-learn


def evaluate_report(y, y_pred):

    clf_report = classification_report(y, y_pred, zero_division=0, digits=4)

    log.info(print(clf_report))


def plot_confusion_matrix(y, hat_y, normalize=False):

    cm = confusion_matrix(y, hat_y)
    ConfusionMatrixDisplay(cm).plot()


# pyhealth


def check_evalu_type(y, hat_y):
    try:
        hat_y = np.array(hat_y).astype(float)
        y = np.array(y).astype(float)
    except:  # noqa E722 # TODO FIX Javier
        raise Exception("not support current data type of hat_y, y")
    _shape_hat_y, _shape_y = np.shape(hat_y), np.shape(y)
    if _shape_hat_y != _shape_y:
        raise Exception("the data shape is not inconformity between y and hey_y")

    label_n_check = set()
    label_item_set = set()
    label_row_set = set()
    for each_y_path in y:
        label_n_check.add(len(np.array(each_y_path)))
        label_item_set.update(np.array(each_y_path).astype(int).tolist())
        label_row_set.add(sum(np.array(each_y_path).astype(int)))

    if len(label_n_check) != 1:
        raise Exception("label_n is inconformity in data")

    if len(label_item_set) <= 1:
        raise Exception("value space size <=1 is unvalid")
    elif len(label_item_set) == 2:
        if 0 in label_item_set and 1 in label_item_set:
            if list(label_n_check)[0] == 1:
                evalu_type = "binaryclass"
            else:
                if max(label_row_set) == 1:
                    evalu_type = "multiclass"
                else:
                    evalu_type = "multilabel"
        else:
            raise Exception("odd value exist in label value space")
    else:
        if list(label_n_check)[0] == 1:
            evalu_type = "regression"
        else:
            raise Exception("odd value exist in label value space")
    return evalu_type


evalu_func_mapping_dict = {
    "binary": binary_eval,
    "multilabel": multilabel_eval,
    "multiclass": multiclass_eval,
    "regression": None,
}


def evaluation(y, hat_y, evalu_type="binary"):
    # evalu_type = label_check(y, hat_y, evalu_type)
    # print ('current data evaluate using {0} evaluation-type'.format(evalu_type))
    evalu_func = evalu_func_mapping_dict[evalu_type]

    return evalu_func(hat_y, y)


# HAIM


def get_ft_pr_tables(y_true, y_pred):
    fpr, tpr, ft_thresholds = roc_curve(y_true, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    auc_value = auc(recall, precision)

    df_result_ft = pd.DataFrame(columns=["fpr", "tpr", "ft_thresholds"])
    df_result_ft["fpr"] = fpr
    df_result_ft["tpr"] = tpr
    df_result_ft["ft_thresholds"] = ft_thresholds

    df_result_pr = pd.DataFrame(columns=["precision", "recall", "pr_thresholds"])
    df_result_pr["precision"] = precision
    df_result_pr["recall"] = recall
    df_result_pr["auc"] = auc_value

    return df_result_pr


# if __name__ == "__main__":
#     y = np.array([0.0, 1.0])
#     hat_y = np.array([[0.3], [0.8]])
#     z = evaluation(hat_y, y)
#     print(z)
#     y = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
#     hat_y = np.array([[0.3, 0.7, 0.1], [0.1, 0.2, 0.8]])
#     z = func(hat_y, y)
#     print(z)
