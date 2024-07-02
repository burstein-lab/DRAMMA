import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import scipy.spatial.distance as ssd
from collections import defaultdict
import scipy.cluster.hierarchy as hc


AA_INDEX_FEATURES = ['ANDN920101', 'ARGP820101', 'ARGP820102', 'BEGF750103', 'BIGC670101', 'BIOV880101', 'BULH740102', 'BUNA790101', 'BURA740101', 'CHAM810101', 'CHOC760102', 'CHOC760103', 'CHOP780202', 'CHOP780205', 'CHOP780213', 'CHOP780214', 'DAYM780101', 'FAUJ880107', 'FAUJ880111', 'FAUJ880112', 'FINA910102', 'GUYH850101', 'ISOY800102', 'ISOY800108', 'KLEP840101', 'LEVM760106', 'NAKH900102', 'OOBM770104', 'PRAM820102', 'PTIO830101', 'QIAN880106', 'RICJ880101', 'VELV850101', 'YUTK870101', 'YUTK870103', 'MUNV940101', 'MUNV940104', 'KUMS000103', 'GEOR030101']
PATTERN = re.compile("[LB]{8}")
PERCENTAGE = 0.025


def get_correlated_features(df, correlation_threshold=0.85):
    corr_df = df.corr(method='pearson').abs()
    corr_df[corr_df.isna()] = 0
    # Select upper triangle of correlation matrix
    upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool)).stack()
    correlated = upper[upper > correlation_threshold].reset_index().sort_values([0], ascending=False).loc[:,
                 ['level_0', 'level_1']]

    correlated = correlated.query('level_0 not in level_1')
    correlated_array = correlated.groupby('level_0').agg(lambda x: chain(x.level_0, x.level_1))
    correlated_array["level_1"] = correlated_array["level_1"].apply(lambda x: list(set(x)))
    return correlated_array.reset_index()['level_1']


def get_correlation_clusters(df, correlation_threshold=0.85):
        corr_df = df.corr(method='pearson').abs()
        corr_df[corr_df.isna()] = 0
        np.fill_diagonal(corr_df.values, 1)  # setting diagonal to 1 to deal with null cols
        ordered_cols = list(corr_df.columns)

        distances = 1 - corr_df.values
        distArray = ssd.squareform(distances)
        hier = hc.linkage(distArray, method="average")
        cluster_labels = hc.fcluster(hier, 1-correlation_threshold, criterion="distance")

        cluster_mapping = defaultdict(list)
        for ind in range(len(cluster_labels)):
            group_num = cluster_labels[ind]
            cluster_mapping[group_num].append(ordered_cols[ind])
        groups = list(cluster_mapping.values())
        return [group for group in groups if len(group) > 1]


def plot_corr_heatmap(cor):
    plt.figure(figsize=(140, 140))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def get_rare_features(df, percentage=PERCENTAGE):
    cols_to_check = list(filter(PATTERN.match, df.columns))  # Taking AA Quality features
    to_check = df[cols_to_check]
    to_check = to_check[to_check > 0].count() / len(df)
    return list(to_check[to_check < percentage].index)


def get_features_to_drop(df, sorted_features, threshold=0.9):
    to_drop = []

    # remove rare features
    rare_features = get_rare_features(df)
    to_drop += rare_features

    # remove correlated features
    df = df.drop('passed threshold', axis=1) if 'passed threshold' in df.columns else df
    corr_features = get_correlation_clusters(df.drop(columns=rare_features), threshold)
    for feature_group in corr_features:
        is_aa_index = [feat in AA_INDEX_FEATURES for feat in feature_group]
        if any(is_aa_index) and not all(is_aa_index):  # we prefer not keep AAindex features if there are other similar features
            to_drop += [feat for feat in feature_group if feat in AA_INDEX_FEATURES]
            feature_group = list(np.array(feature_group)[~np.array(is_aa_index)])
        best_feature_ind = np.argmin([sorted_features.index(feature) for feature in feature_group])
        feature_group.pop(best_feature_ind)
        to_drop += feature_group

    return to_drop


