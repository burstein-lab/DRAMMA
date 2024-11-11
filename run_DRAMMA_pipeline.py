import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import os
import sys
from run_model import get_model_results_single_label, get_model_results_multi_class
from feature_extraction.run_features import create_features_from_dir
from feature_extraction.feature_list import FeatureList, ALL_FEATURES
from utilities import combine_all_pkls
from feature_extraction.train_dataset_creator import run_dataset_creator, fix_df, COLUMNS


def get_model_res(model_pickle, dataset_path, is_multi_class):
    if is_multi_class:
        model_res = get_model_results_multi_class(model_pickle, dataset_path)
    else:
        model_res = get_model_results_single_label(dataset_path, pkl=model_pickle, filter_pos=False, filter_low_scores=False)[['Label', 'prob', 'passed_0.75', 'passed_0.95']]
    return model_res


def single_sample_pipeline(args, feature_lst):
    # Running Feature extraction and saving it to file
    feature_lst.run_features(*args.dif_format_paths, out_dir=args.feature_dir)
    feature_dir = os.path.join(args.feature_dir, Path(Path(args.dif_format_paths[0]).stem).stem)
    df = combine_all_pkls(feature_dir, pd.DataFrame([]).rename_axis('ID'))
    df = fix_df(df, False, COLUMNS)
    if not os.path.exists(os.path.join(args.feature_dir, 'united_dataset')):
        os.makedirs(os.path.join(args.feature_dir, 'united_dataset'))
    dataset_path = os.path.join(args.feature_dir, 'united_dataset', 'all_features.pkl')
    df.to_pickle(dataset_path)

    model_res = get_model_res(args.model, dataset_path, args.multi_class)

    if not args.keep_files:
        os.system(f"rm -r {os.path.join(args.feature_dir, 'united_dataset')}")
        os.system(f"rm -r {feature_dir}")
    return model_res


def multi_sample_pipeline(args, feature_lst):
    # making sure the feature_dir is specifically for features
    feature_dir = args.feature_dir if 'features' in os.path.split(args.feature_dir)[-1] else os.path.join(args.feature_dir, 'features')
    create_features_from_dir(args.input_path, feature_lst, args.suffix, feature_dir)
    united_datasets_path = os.path.join(args.feature_dir, 'united_dataset')
    if not os.path.exists(united_datasets_path):
        os.makedirs(united_datasets_path)
    run_dataset_creator(feature_dir, whitelist='', is_pumps=False, columns=COLUMNS, all_data=True, is_pickle=True, batch_size=args.batch_size, out_dir=united_datasets_path)
    results = []
    for entry in os.listdir(united_datasets_path):  # Going over all batch files
        dataset_path = os.path.join(united_datasets_path, entry)
        model_res = get_model_res(args.model, dataset_path, args.multi_class)
        results.append(model_res)

    if not args.keep_files:
        os.system(f"rm -r {united_datasets_path}")
        os.system(f"rm -r {feature_dir}")
    return pd.concat(results)


def main(args):
    features_to_run = [feat for feat in ALL_FEATURES if feat not in args.features_to_drop]
    feature_lst = FeatureList(args.hmmer_path, args.mmseqs_path, args.tmhmm_path, args.kmer, True,
                              args.label_threshold, args.threshold_list, args.gene_window, args.nucleotide_window,
                              features=features_to_run, n_cpus=args.ncpus)
    if args.dif_format_paths:
        res = single_sample_pipeline(args, feature_lst)
    else:
        res = multi_sample_pipeline(args, feature_lst)

    res.to_pickle(args.output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run feature extraction on a given sample or a directory of samples, and run trained the model on them.')
    # feature extraction
    parser.add_argument('--input_path', type=str, help='Insert the full path of the wanted directory with all the assemblies. not needed if --dif_format_paths is supplied.')
    parser.add_argument("--dif_format_paths", nargs='*', type=str,
                        help="The data in the 4 different formats faa,gff,ffn,fa. if we want to only run the pipeline on one sample. if supplied --input_path is not needed.")
    parser.add_argument('--hmmer_path', type=str, help="full path to the HMMER's hmmsearch program.")
    parser.add_argument('--mmseqs_path', type=str, help='full path to the Mmseqs2 program.')
    parser.add_argument('--tmhmm_path', type=str, help='full path to the tmhmm program.')
    parser.add_argument('--feature_dir', type=str, help='the path to the directory we want to save our features in, default: "features" (new sub directory of current directory)', default='features')
    parser.add_argument("-k", "--kmer", type=int, default=4, help="It will run the kmers count from 2 to k, default=4")
    parser.add_argument("-lt", "--label_threshold", type=str, default="1e-10",
                        help="Threshold for the proximity feature -  hmm comparison, default=0.000001")
    parser.add_argument("-t", "--threshold_list", nargs='*', type=str, default=['1e-8'],
                        help="Threshold for the proximity feature -  hmm comparison, default=['1e-8']")
    parser.add_argument("-d", "--gene_window", type=int, default=10, help="Size of the ORFs window, default=10")
    parser.add_argument("-n", "--nucleotide_window", type=int, default=10000, help="Size of the nucleotides window, default=10000")
    parser.add_argument("--ncpus", type=int, default=64, help="Number of cpus to use for feature extraction, default=64")
    parser.add_argument("-sf", "--suffix", default='', help="suffix to sample files such that the protein file will end with {suffix}proteins.faa."
                                                                    " for example, .min10k. to get only contigs of length more than 10k. Input '' (default) if none applies")
    parser.add_argument("-ftd", "--features_to_drop", nargs='*', type=str, default=['CARD_labeling', 'DNA_KMers', 'Cross_Membrane', 'GC_Content', 'SmartGC'],
                        help="The list of features names (according to the names in FeaturesList) that we dont want"
                             "to execute. default: ['CARD_labeling', 'DNA_KMers', 'Cross_Membrane', 'GC_Content', 'SmartGC']")
    # dataset creation
    parser.add_argument("-b", "--batch_size", type=int, default=0, help="batch size for saving dataset for all_data=True. default: 0 (everything will be saved in one file)")
    # Running model
    parser.add_argument("--model", default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "models", "DRAMMA_AMR_model.pkl"), type=str, help="path to pickle with model, relevant cols and thresholds dict. default: ./data/models/DRAMMA_AMR_model.pkl")
    parser.add_argument("-mc", '--multi_class', dest='multi_class', action='store_true', help='Choose this to run multi_class models. if multi-label is True, this param is irrelevant. default: False (single_class)')
    parser.add_argument("-sc", '--single_class', dest='multi_class', action='store_false', help='Choose this to run single_class model. if multi-label is True, this param is irrelevant. default: False (single_class)')
    # Cleaning up
    parser.add_argument("-out", "--output_file", type=str, help="path to pkl file we want to save our results in")
    parser.add_argument('--keep_files', dest='keep_files', action='store_true', help='Choose this to keep feature files. Those files can be reused'
                             'the next time this script is executed, default: False (remove_files)')
    parser.add_argument('--remove_files', dest='keep_files', action='store_false', help='Choose this to remove all feature files. default: False (remove_files)')
    parser.set_defaults(keep_files=False)
    parser.set_defaults(multi_class=False)
    args = parser.parse_args()
    main(args)
