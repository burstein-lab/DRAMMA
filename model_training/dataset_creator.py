import pandas as pd
import numpy as np
from .AMR_model import AMRModel
from utilities import parse_query_name, parse_arg_name
try:
    from sklearn.model_selection import StratifiedGroupKFold as GroupKFold
except ImportError:
    print("Could not import StratifiedGroupKFold")
    from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle


COLS_TO_REMOVE = ['query_name', 'query_accession', 'best_eval', 'best_eval_exponent', 'best_score', 'passed threshold']
NEG_SIZE = 10


def load_df(file_path):
    if ".pkl" in file_path:
        df = pd.read_pickle(file_path)
    elif ".tsv.gz" in file_path:
        df = pd.read_csv(file_path, sep='\t', compression="gzip")
    else:
        df = pd.read_csv(file_path, sep='\t')
    return df.drop_duplicates()


def get_binary_amr_labels(data):
    factor = pd.factorize(data['passed threshold'].fillna(False))
    data['Label'] = factor[0]
    if factor[1][0] == True:  # if resistance label is zero
        data['Label'] = data['Label'].apply(lambda x: 1 - x)
    return factor[1]


def get_multi_amr_labels(data, label_dict, label_lst=()):
    data['query_name'] = data['query_name'].apply(parse_query_name).map(label_dict)
    # Making sure the first value is False, so it will have a label of 0
    overhead = [False] + [lb for lb in label_lst if lb != False and lb != 'Non-AMR']  # keeping order of the given labels
    factor = pd.factorize(overhead + list(data.loc[:, 'query_name'].fillna(False)))
    data['Label'] = factor[0][len(overhead):]
    return factor[1]


def get_dataset(file_path, label_dict: dict = None, drop_na=True, label_lst=()):
    data = load_df(file_path)
    unique_labels = get_multi_amr_labels(data, label_dict, label_lst) if label_dict else get_binary_amr_labels(data)
    data.drop(COLS_TO_REMOVE, axis=1, inplace=True, errors='ignore')
    if drop_na: # drop empty columns
        data.dropna(how="all", axis=1, inplace=True)

    if "ID" in data.columns:
        data.set_index('ID', inplace=True)

    # removing inf values
    data.replace(np.inf, np.finfo(np.float32).max, inplace=True)
    data.replace(-np.inf, np.finfo(np.float32).min, inplace=True)
    return data, unique_labels


def get_test_df(file_path, new_labels_file=None, label_column_name=None, label_lst=(), drop_zero=False):
    if new_labels_file:
        df, _ = get_multi_class_dataset(file_path, new_labels_file, label_column_name, balance=False, label_lst=label_lst, drop_zero=drop_zero)
    else:
        df, _ = get_dataset(file_path, drop_na=False)
    X, y = df.drop(columns=['Label']), df['Label']
    return X, y


def cv_custom_splitter(n_splits, df, split_by='Contig'):
    '''
    splits the samples based on the column provided by split by
    :param n_splits: int. number of time we will need to split,
    :param df: dataframe with all the samples, with 'split_by' as the seperating column
    :param split_by: column to split the data on. default is contig so there is no data leakage from the genes on the same contig.
    :return: yields train and test.
    '''

    grps = df[split_by]
    try:
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    except TypeError:
        print("GroupKFold is not stratified, please update sklearn!")
        gkf = GroupKFold(n_splits=n_splits)
    X, y = split_the_data(df)

    for train, test in gkf.split(X, y, groups=grps):
        X_train, y_train = split_the_data(df.iloc[train, :])
        X_test, y_test = split_the_data(df.iloc[test, :])
        yield X_train, X_test, y_train, y_test


def split_the_data(gene_df):
    X = gene_df.drop('Label', axis=1)
    y = gene_df.loc[:, 'Label'].copy().astype('int')
    return X, y


def split_by_id_terms(df, terms_for_test_ids):
    mask = df.index.str.contains(terms_for_test_ids)
    df_train, df_test = df[~mask], df[mask]
    X_train, y_train = split_the_data(df_train)
    X_test, y_test = split_the_data(df_test)
    return X_train, X_test, y_train, y_test


def get_multi_class_dataset(data_file, labels_file, label_col_name, balance=True, drop_zero=False, label_lst=()):
    new_labels = pd.read_csv(labels_file, sep='\t')
    new_labels['ARG Name'] = new_labels['ARG Name'].apply(parse_arg_name)
    labels_dict = new_labels.set_index('ARG Name')[label_col_name].to_dict()
    df, label_indexes = get_dataset(data_file, labels_dict, label_lst=label_lst)
    if balance: # we are making sure zero class is equal to smallest class
        neg_size = df['Label'].value_counts().min()
        print(neg_size)
        neg = df[df['Label'] == 0].sample(n=neg_size)
        df = shuffle(pd.concat([df[df['Label'] != 0], neg]))
    if drop_zero:
        df = df[df['Label'] != 0]
        df['Label'] = df['Label'] - 1  # Reducing Label indices in 1
        label_indexes = label_indexes[1:]  # Removing zero from label list
    df.replace(np.inf, np.finfo(np.float32).max, inplace=True)
    df.replace(-np.inf, np.finfo(np.float32).min, inplace=True)
    return df, label_indexes


def get_split_dataset(data_file, new_labels_file=None, label_column_name=None, split_by='Contig', get_cv=False, features=(), n_feats=0, param_dict=None, n_jobs=2, drop_zero=False, cluster_file=''):
    if new_labels_file:
        df, label_indexes = get_multi_class_dataset(data_file, new_labels_file, label_column_name, drop_zero=drop_zero)
    else:
        df, label_indexes = get_dataset(data_file)

    if not isinstance(split_by, str):  # there's a list for terms in id to be in test set
        X_train, X_test, y_train, y_test = split_by_id_terms(df, split_by)
    else:
        if split_by == 'Cluster':
            df['Cluster'] = df.index.map(pd.read_csv(cluster_file, sep='\t', names=['rep', 'ID']).set_index('ID')['rep'].to_dict())
        cvs = []
        df = df[pd.notnull(df[split_by])]
        for X_train, X_test, y_train, y_test in cv_custom_splitter(5, df, split_by):
            fold_label = "" if split_by in ('Contig', 'Cluster') else X_test.iloc[0][split_by]
            X_train, X_test = X_train.drop(list({'Contig', split_by}), axis=1), X_test.drop(list({'Contig', split_by}), axis=1)
            curr_model = AMRModel(X_train, y_train, features, n_feats, n_jobs, param_dict)  # if features is empty, calculates best features.
            if get_cv:
                cvs.append((curr_model, X_test[curr_model.features], y_test, fold_label))
            else:
                break
    if not get_cv or not isinstance(split_by, str):
        return curr_model, X_train[curr_model.features], X_test[curr_model.features], y_test, list(label_indexes)
    else:
        return cvs, list(label_indexes)

