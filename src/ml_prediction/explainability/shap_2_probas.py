"""
SHAP to Probabilities. Binary Classifier
Javier Monreal Tolmo
TODO: Add docstrings (JAVIER)
"""

# import matplotlib.pyplot as plt


# def shap2deltaprob(features, shap_df, shap_sum, probas, func_shap2probas):
#     """
#     map shap to Δ probabilities
#     --- input ---
#     :features: list of strings, names of features
#     :shap_df: pd.DataFrame, dataframe containing shap values
#     :shap_sum: pd.Series, series containing shap sum for each observation
#     :probas: pd.Series, series containing predicted probability for each observation
#     :func_shap2probas: function, maps shap to probability (for example interpolation function)
#     --- output ---
#     :out: pd.Series or pd.DataFrame, Δ probability for each shap value
#     """
#     # 1 feature
#     if type(features) == str or len(features) == 1:
#         return probas - (shap_sum - shap_df[features]).apply(func_shap2probas)
#     # more than 1 feature
#     else:
#         return shap_df[features].apply(lambda x: shap_sum - x).apply(func_shap2probas).apply(lambda x: probas - x)


# def partial_deltaprob(feature, X, shap_df, shap_sum, probas, func_shap2probas, cutoffs=None):
#     """
#     return univariate analysis (count, mean and standard deviation) of shap values based on the original feature
#     --- input ---
#     :feature: str, name of feature
#     :X: pd.Dataframe, shape (N, P)
#     :shap_df: pd.DataFrame, shape (N, P)
#     :shap_sum: pd.Series, series containing shap sum for each observation
#     :probas: pd.Series, series containing predicted probability for each observation
#     :func_shap2probas: function, maps shap to probability (for example interpolation function)
#     :cutoffs: list of floats, cutoffs for numerical features
#     --- output ---
#     :out: pd.DataFrame, shape (n_levels, 3)
#     """
#     dp_col = shap2deltaprob(feature, shap_df, shap_sum, probas, func_shap2probas)
#     dp_col_mean = dp_col.mean()
#     dp_col.name = "DP_" + feature
#     out = pd.concat([X[feature], dp_col], axis=1)
#     if cutoffs:
#         intervals = pd.IntervalIndex.from_tuples(list(zip(cutoffs[:-1], cutoffs[1:])))
#         out[feature] = pd.cut(out[feature], bins=intervals)
#         out = out.dropna()
#     out = out.groupby(feature).describe().iloc[:, :3]
#     out.columns = ["count", "mean", "std"]
#     out["std"] = out["std"].fillna(0)

#     return out

#     # Shap on Catboost


# def shap_catboost(model, X, categorical_features):

#     shap_df = model.get_feature_importance(data=Pool(X, cat_features=categorical_features), type="ShapValues")
#     shap_df = pd.DataFrame(shap_df[:, :-1], columns=X.columns, index=X.index)
#     shap_sum = shap_df.sum(axis=1)

#     # build interpolation function to map shap into probability
#     shap_sum_sort = shap_sum.sort_values()
#     probas_cat_sort = probas_cat[shap_sum_sort.index]

#     intp = interp1d(shap_sum_sort, probas_cat_sort, bounds_error=False, fill_value=(0, 1))

#     return shap_sum_sort, probas_cat_sort


# def show_shap_df(model, X, categorical_features):

#     # show shap
#     shap_df = model.get_feature_importance(data=Pool(X, cat_features=categorical_features), type="ShapValues")

#     temp = shap_df.head().round(2)
#     temp.style.apply(lambda x: ["background:orangered" if v < 0 else "background:lightgreen" for v in x], axis=1)

#     # show Δ probabilities
#     temp = (
#         shap2deltaprob(X.columns.to_list(), shap_df, shap_sum, probas_cat, func_shap2probas=intp)
#         .head()
#         .applymap(lambda x: ("+" if x > 0 else "") + str(round(x * 100, 2)) + "%")
#     )

#     temp.style.apply(
#         lambda x: ["background:orangered" if float(v[:-1]) < 0 else "background:lightgreen" for v in x], axis=1
#     )


# def plot_shap_probabilities(shap_sum_sort, probas_cat_sort):

#     plt.scatter(
#         0,
#         intp(0),
#         s=200,
#         fc="red",
#         ec="black",
#         zorder=3,
#         label="Observation considering all features\nexcept Passenger Age",
#     )
#     plt.scatter(2, intp(2), s=200, fc="lime", ec="black", zorder=3, label="Observation considering all features")

#     plt.annotate(
#         s="", xy=(2 - 0.5, intp(2) + 0.5 * 0.18), xytext=(0 - 0.5, intp(0) + 0.5 * 0.18), arrowprops={"fc": "black"}
#     )
#     plt.text(-1.5, 0.72, "Marginal effect of\nPassenger Age", ha="center", va="center", fontsize=13)

#     plt.annotate(s="", xy=(2, intp(0) - 0.5 * 0.18), xytext=(0, intp(0) - 0.5 * 0.18), arrowprops={"fc": "black"})
#     plt.text(1, 0.23, "Δ Shap sum\n(= +2)", ha="center", va="top", fontsize=13)

#     plt.annotate(s="", xy=(2 + 0.5, intp(2)), xytext=(2 + 0.5, intp(0)), arrowprops={"fc": "black"})
#     plt.text(4, (intp(0) + intp(2)) / 2, "Δ Predicted\n Probability\n(= +44%)", va="center", ha="center", fontsize=13)

#     legend = plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", fontsize=13)
#     frame = legend.get_frame()
#     frame.set_facecolor("lightyellow")
#     frame.set_edgecolor("black")

#     plt.scatter(shap_sum_sort, probas_cat_sort, color="blue", zorder=2)

#     plt.gca().set_facecolor("lightyellow")
#     plt.grid(zorder=1)
#     plt.xlabel("SHAP sum", fontsize=13)
#     plt.ylabel("Predicted Probability\nof Survival", fontsize=13)
#     plt.xticks(fontsize=13)
#     plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=13)
#     plt.title("From SHAP to Predicted Probability", fontsize=15)

#     plt.savefig("shap2probability_example.png", bbox_inches="tight", dpi=300)


# def plot_partial_deltaprob(X_all, feature, shap_df, shap_sum, probas_cat, intp):

#     dp = partial_deltaprob(feature, X_all, shap_df, shap_sum, probas_cat, func_shap2probas=intp)

#     plt.plot([0, len(dp) - 1], [0, 0], color="dimgray", zorder=3)
#     plt.plot(range(len(dp)), dp["mean"], color="red", linewidth=3, label="Avg effect", zorder=4)
#     plt.fill_between(
#         range(len(dp)),
#         dp["mean"] + dp["std"],
#         dp["mean"] - dp["std"],
#         color="lightskyblue",
#         label="Avg effect +- StDev",
#         zorder=1,
#     )
#     yticks = list(np.arange(-0.2, 0.41, 0.1))
#     plt.yticks(yticks, [("+" if y > 0 else "") + f"{y:.1%}" for y in yticks], fontsize=13)
#     plt.xticks(range(len(dp)), dp.index, fontsize=13)
#     plt.ylabel("Effect on Predicted\nProbability of Survival", fontsize=13)
#     plt.xlabel(feature, fontsize=13)
#     plt.title("Marginal effect of\nSex", fontsize=15)
#     plt.gca().set_facecolor("lightyellow")
#     plt.grid(zorder=2)
