import argparse
import pandas as pd
import numpy as np
import pickle
import os
from model_training.dataset_creator import get_test_df

LABEL_FILE = os.path.join('data', 'arg_mech_drugs.tsv')


def get_result_df(model, rel_cols, X_test, y_test, thresholds_dict, filter_pos=True, filter_low_scores=True):
    test_probabilities = model.predict_proba(X_test[rel_cols])[:, 1]
    result_df = pd.concat([X_test.iloc[:, 0], y_test], axis=1)  # getting id from X_test
    result_df['prob'] = test_probabilities
    result_df = result_df[['Label', 'prob']]
    min_pre = min(thresholds_dict.keys())
    result_df = result_df[result_df['Label'] == 0] if filter_pos else result_df

    if filter_low_scores:
        min_threshold = thresholds_dict.pop(min_pre)
        result_df = result_df[result_df["prob"] >= min_threshold]

    for pre, thresh in thresholds_dict.items():
        result_df[f'passed_{round(pre, 2)}'] = result_df["prob"] >= thresh

    return result_df


def get_model_results_single_label(input_file, pkl, filter_pos=True, filter_low_scores=True):
    with open(pkl, "rb") as f_in:
        data_objs = [pickle.load(f_in) for i in range(2)]
        model_obj, threshold_dict = data_objs
        model, rel_cols = model_obj.model, model_obj.features

    X_test, y_test = get_test_df(input_file)
    res_df = get_result_df(model, rel_cols, X_test, y_test, threshold_dict, filter_pos=filter_pos, filter_low_scores=filter_low_scores)
    return res_df


def get_model_results_multi_class(pkl, input_file, label_col="Updated Resistance Mechanism"):
    with open(pkl, "rb") as f_in:
        data_objs = [pickle.load(f_in) for i in range(2)]
        model_obj, classes = data_objs
        model, rel_cols = model_obj.model, model_obj.features
        vfunc = np.vectorize(lambda x: classes[x])

    X_test, y_test = get_test_df(input_file, new_labels_file=LABEL_FILE, label_column_name=label_col)
    result_df = X_test.iloc[:, 0:2]  # getting id from X_test
    all_probs = model.predict_proba(X_test[rel_cols])
    result_df.loc[:, label_col + "_class"] = vfunc(np.argmax(all_probs, axis=1))
    result_df.loc[:, label_col + "_prob"] = np.max(all_probs, axis=1)
    return result_df[[label_col + "_class", label_col + "_prob"]]


def main(args):
    if args.multi_class:
        res_df = get_model_results_multi_class(args.pickle, args.input_file)
    else:
        res_df = get_model_results_single_label(args.input_file, args.pickle, args.filter_pos, args.filter_low_scores)
    res_df.to_pickle(args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run given model on the input file and return results of proteins that passed the given threshold')
    parser.add_argument("-pkl", "--pickle", type=str, help="path to pickle with model, relevant cols and thresholds dict.")
    parser.add_argument("-in", "--input_file", type=str, help="path to feature file we want to run the model against")
    parser.add_argument("-out", "--output_file", type=str, help="path to pkl file we want to save our results in")
    parser.add_argument("-fp", '--filter_pos', dest='filter_pos', action='store_true', help='Choose this to keep only negative proteins, default: True (filter_pos)')
    parser.add_argument("-kp", '--keep_pos', dest='filter_pos', action='store_false', help='Choose this to keep both positive and negative proteins, default: False (filter_pos)')
    parser.add_argument("-fp", '--filter_low_scores', dest='filter_low_scores', action='store_true', help='Choose this to only keep proteins that passed the minimal model score, default: True (filter_low_scores)')
    parser.add_argument("-kp", '--keep_low_scores', dest='filter_low_scores', action='store_false', help='Choose this to keep results of proteins that received low model scores, default: False (filter_low_scores)')
    parser.add_argument("-mc", '--multi_class', dest='multi_class', action='store_true', help='Choose this to run multi_class models. default: False (single_class)')
    parser.add_argument("-sc", '--single_class', dest='multi_class', action='store_false', help='Choose this to run single_class model.  default: False (single_class)')
    parser.set_defaults(filter_pos=True)
    parser.set_defaults(filter_low_scores=True)
    parser.set_defaults(multi_class=False)
    args = parser.parse_args()
    main(args)
