"""
TODO: Add docstrings (JAVIER)
"""

# import time
# import numpy as np
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score  # , train_test_split

# def test_model(model, X_test, y_test, scoring: str | list = "accuracy"):
#     """
#     Get the RMSE for a given model on a test dataset

#     Parameters
#     ----------
#     model: a model implementing the standard scikit-learn interface
#     X_test: pd.DataFrame holding the features of the test set
#     y_test: pd.Series holding the test set target

#     Returns
#     -------
#     test_score: the RMSE on the test dataset
#     """

#     predictions = model.predict(X_test)
#     test_score = mape(y_test, predictions)
#     return test_score


def eval_model(model, X_train, y_train, scoring: str | list = "f1_macro"):

    # cross validate using the right iterator for time series
    # """
    # fit_params={#'early_stopping_rounds': 30,
    #         'eval_metric': 'rmse',
    #         'verbose': -1,
    #         'eval_set': [[X_test, y_test]],
    #         'callbacks':[pruning_callback]}
    # """
    cv_space = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)  # StratifiedKFold(n_splits=5)

    # t1 = time.clock()
    cv_score = cross_val_score(model, X_train, y_train, cv=cv_space, scoring=scoring)
    # t2 = time.clock()

    score = (cv_score.mean(), cv_score.std())
    print(f" {cv_score.mean():.2f} {scoring} with a standard deviation of {cv_score.std():.2f}")

    return score
