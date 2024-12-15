from dataset_creator import get_split_dataset
from model_evaluation.evaluate_model import get_model_quality_stats
from feature_importance import get_rf_default_importance
import pandas as pd


def get_train_analysis(train_file, selected_features, label_file="", label_col='Updated Resistance Mechanism', split_by='Contig', n_feats=0, save_figs=False, n_jobs=2, param_dict=None, drop_zero=False, cluster_file=''):
    cvs, label_names = get_split_dataset(train_file, get_cv=True, features=selected_features, n_feats=n_feats, new_labels_file=label_file, label_column_name=label_col, split_by=split_by, param_dict=param_dict, n_jobs=n_jobs, drop_zero=drop_zero, cluster_file=cluster_file)
    if not drop_zero:
        label_names[0] = 'Non-AMR'

    if not label_file:
        # Feature Selection
        get_rf_default_importance(cvs, n=6, display=True, save_figs=save_figs)

    # Quality
    prefix = '' if not label_file else label_col.replace(" ", "_")
    label_names = label_names if label_file else ()
    _, roc, aupr, thresholds = get_model_quality_stats(None, cvs, None, None, save_figs=save_figs, prefix=prefix, label_names=label_names)  # no need for those parameters in cvs
    return thresholds


def process_results(X_test, all_probs, all_pred, y_test, ds):
    result_df = pd.concat([X_test.iloc[:, 0], y_test], axis=1)
    result_df['prob'] = all_probs
    result_df['pred'] = all_pred
    result_df = result_df[['Label', 'prob', 'pred']]
    return result_df.sort_values(by=['prob'], ascending=False).reset_index()
