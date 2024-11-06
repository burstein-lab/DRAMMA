import matplotlib as mpl
mpl.use('Agg')
from AMR_model import AMRModel
from model_training import get_train_analysis
from model_evaluation.evaluate_model import get_model_quality_stats
from utilities import load_param_pkl
from dataset_creator import get_test_df, get_dataset, split_the_data, get_multi_class_dataset
import argparse
import numpy as np
import os
import pickle
import json

COLS_TO_REMOVE = ['query_name', 'query_accession', 'best_eval', 'best_eval_exponent', 'best_score', 'passed threshold', 'Contig']
LABEL_FILE = os.path.join('..', 'data', 'arg_mech_drugs.tsv')
TMP_FILE = os.path.join(os.getcwd(), "create_pickle_tmp.txt")


def log_stats(txt):
    with open(TMP_FILE, 'w') as fout:
        fout.write(txt)


def main(args, param_dict, selected_feats):
    if args.is_multiclass:
        threshold_dict = get_train_analysis(args.input_path, selected_feats, label_file=LABEL_FILE,
                                            label_col=args.label_col, n_feats=args.n, save_figs=True, n_jobs=args.n_jobs,
                                            param_dict=param_dict, drop_zero=True)
        log_stats("Finished get_train_analysis")
        for class_num, t_dict in threshold_dict.items():
            for key in list(t_dict.keys()):
                t_dict[round(key, 3)] = np.median(t_dict.pop(key))
        log_stats("got thresholds")
        dataset, labels = get_multi_class_dataset(args.input_path, LABEL_FILE, args.label_col, drop_zero=True)
        X_train, y_train = split_the_data(dataset)
        amr_model = AMRModel(X_train, y_train, selected_feats, n_feats=args.n, n_jobs=args.n_jobs, param_dict=param_dict)
        log_stats("Finished model training")
        data_objs = [amr_model.model, amr_model.features, labels, threshold_dict]

        if args.test_set:
            X_test, y_test = get_test_df(args.test_set, new_labels_file=LABEL_FILE, label_column_name=args.label_col, label_lst=labels, drop_zero=True)
            crosstab, _, _, test_threshold_dict = get_model_quality_stats(amr_model.model, [], X_test[amr_model.features], y_test, save_figs=True, display=True, prefix=f"{args.label_col.replace(' ', '_')}_multiclass_", label_names=labels)
            log_stats("Finished test eval")
            print(crosstab)

    else:
        # This function also creates feature importance fig and 5-fold AUPR and AUROC figures
        threshold_dict = get_train_analysis(args.input_path, selected_feats, label_file="", save_figs=True, n_feats=args.n, n_jobs=args.n_jobs, param_dict=param_dict)
        log_stats("Finished get_train_analysis")
        for key in list(threshold_dict.keys()):
            threshold_dict[round(key, 3)] = np.median(threshold_dict.pop(key))
        log_stats("got thresholds")

        X_train, y_train = split_the_data(get_dataset(args.input_path)[0])
        amr_model = AMRModel(X_train, y_train, selected_feats, n_feats=args.n, n_jobs=args.n_jobs, param_dict=param_dict)
        log_stats("Finished model training")
        data_objs = [amr_model.model, amr_model.features, threshold_dict]

        if args.test_set:
            X_test, y_test = get_test_df(args.test_set)
            crosstab, _, _, test_threshold_dict = get_model_quality_stats(amr_model.model, [], X_test[amr_model.features], y_test, save_figs=True, display=True)
            log_stats("Finished test eval")
            print(crosstab)

    with open(args.output_path, "wb") as f_out:
        for obj in data_objs:
            pickle.dump(obj, f_out)


def parse_bool_arg(arg):
    return arg in ["True", "true", "T", "t"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation a pkl file with all the relevant information for running the model')
    parser.add_argument("-i", "--input_path", type=str, help="Path to input dataset")
    parser.add_argument("-o", "--output_path", type=str, help="Path to output pickle")
    parser.add_argument("-ts", "--test_set", type=str, help="Path to test set, if we want to evaluate model. default: '' (no evaluation)", default="")
    parser.add_argument("-nj", "--n_jobs", type=int, help="number of jobs to use to train the model. default:2", default=2)
    parser.add_argument("--param_pkl", help="Path to pickle with chosen hyperparams. if equals to '', Random Forest's default params are used. "
                                            "default='data/hyper_param_opt.pkl'", type=str, default=os.path.join('data', 'hyper_param_opt.pkl'))
    parser.add_argument("--feature_json", help="Path to json with chosen features. if no file is supplied (''),"
                                               " the best n features are calculted are taken."
                                               " default='data/DRAMMA_selected_features.json'",
                        type=str, default=os.path.join('data', 'DRAMMA_selected_features.json'))
    parser.add_argument("-n", "--n", type=int, default=30, help="how many top features to choose. if 0 - chooses all features not correlated. default: 30")
    parser.add_argument("-mc", "--is_multiclass", type=str, default="False", help="should we create a model for multiclass classifier")
    parser.add_argument("-lc", "--label_col", help="What column of LABEL_FILE to take for labeling. "
                                                   "default: Updated Resistance Mechanism",
                        type=str, default='Updated Resistance Mechanism')
    args = parser.parse_args()
    args.is_multiclass = parse_bool_arg(args.is_multiclass)
    param_dict = load_param_pkl(args.param_pkl)
    selected_feats = []
    if args.feature_json:
        with open(args.feature_json, 'r') as fin:
            selected_feats = json.load(fin)
    main(args, param_dict, selected_feats)