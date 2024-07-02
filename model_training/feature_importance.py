import matplotlib.pyplot as plt
from model_evaluation.feature_stats import plot_attributes_distributions
from utilities import parse_col_name
from collections import Counter
import pickle
import pandas as pd
import os


DEFAULT_IMAGE_PREFIX = os.path.join(os.getcwd(), "model_score_by_feature_num")


def get_rf_default_importance(cvs, top_features=30, n=5, display=True, save_figs=False, per_cv=False):
    """
    Shows a plot with the 20 best features according to Random Forest Default Importance
    (mean and standard deviation of accumulation of the impurity decrease within each tree) and prints them,
    and plots the distributions of the n best features in class 0 and 1
    """
    importances = []
    ind = 0
    with open(os.path.join(os.getcwd(), "tmp_rf_default_importance.pkl"), 'wb') as fout:
        for amr_model, X_test, y_test, _ in cvs:
            rf = amr_model.model.steps[1][1]
            imps = pd.Series(rf.feature_importances_, index=amr_model.features)
            pickle.dump(imps, fout)
            importances.append(imps)
            ind += 1
    importances = importances if per_cv else pd.concat(importances, axis=1).mean(axis=1)
    if display and not per_cv:  # currently not supporting plot per cv
        best_feats = importances.sort_values(ascending=False).iloc[:top_features].sort_values()
        plt.figure(figsize=(20, 10))
        plt.title(f"Top {top_features} Features According to Random Forest's Default Feature Importance", fontsize=22, pad=12)
        plt.barh(range(len(best_feats)), best_feats.values, color='#7DE1FC', align='center')
        plt.xticks(fontsize=20)
        plt.yticks(range(len(best_feats)), list(best_feats.index.map(parse_col_name)), fontsize=20)
        plt.xlabel('Relative Importance', fontsize=22, labelpad=10)
        plt.tight_layout()
        plt.savefig(f"RF__default_importance_top_{top_features}.pdf", bbox_inches='tight') if save_figs else plt.show()

    if per_cv:
        sorted_features = [list(fdf.sort_values(ascending=False).index) for fdf in importances]
    else:
        sorted_features = list(importances.sort_values(ascending=False).index)
    if display and not per_cv:  # currently not supporting plot per cv
        X, y = pd.concat([cv[1] for cv in cvs]), pd.concat([cv[2] for cv in cvs])
        plot_attributes_distributions(X, y, sorted_features, n=n, save_figs=save_figs)
        print(f"Feature importance - Top {top_features}:")
        print(list(sorted_features[:top_features]))
    return sorted_features


def get_features_by_count(feature_dict, count, wanted_len=0):
    counter = Counter()
    wanted_len = len(feature_dict['SelectFromModel']) if wanted_len == 0 else wanted_len
    for key, value in feature_dict.items():
        if key == 'RF_default':  # selectFromModel is taken from RF default importance
            continue
        counter.update(value[:min(len(value), wanted_len)])

    selected_features = [k for k, v in counter.items() if v >= count]
    return selected_features
