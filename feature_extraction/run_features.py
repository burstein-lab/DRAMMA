import warnings
warnings.filterwarnings('ignore')
import argparse
import time
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import go_through_files
from feature_extraction.feature_list import FeatureList, ALL_FEATURES


def create_features_pkls(feature_lst, dir_path, files_paths, outdir):
    """
    creates pkl for each sample with all the features.
    NOTICE: if the one of the features does not run, the function will pass it and continue to the next sample
    :param feature_lst: A Featurelist object with all the wanted feature classes
    :param dir_path: the path of the sample
    :param files_paths: list, contains: proteins_fasta, gff, ffn, fa
    :param outdir: path to directory we want to save our features in.
    :return: Does not return anything. Creates pkl files in the current directory
    """
    all_start = time.time()
    if not dir_path or dir_path in files_paths[0]:  # files already have full path
        feature_lst.run_features(f'{files_paths[0]}', f'{files_paths[1]}', f'{files_paths[2]}', f'{files_paths[3]}', out_dir=outdir, do_print=True)
    else:
        feature_lst.run_features(os.path.join(dir_path, files_paths[0]), os.path.join(dir_path, files_paths[1]),
                     {os.path.join(dir_path, files_paths[2])}, os.path.join(dir_path, files_paths[3]), out_dir=outdir, do_print=True)

    print('finished all the features individually')
    all_end = time.time()
    print("Time consumed in working on it all: ", all_end - all_start)


def create_features_from_dir(input_path, feature_lst, suffix, output_dir):
    """
    goes through all the sample files and run all the features on each sample
    :param input_path: starts with the input_path given by the user, keep entering the folders
    :param feature_lst: object for the calculation of the relevant features
    :param suffix: a prefix to the protein sample files you want to get, such as the protein file will end with {suffix}proteins.faa
    :param output_dir: path to directory we want to save our features in.
    :return: for each sample, df wrapped as pkl file, in the folder "all_merged_pkls"
    """
    for files_paths in go_through_files(input_path, suffix):
        create_features_pkls(feature_lst, input_path, files_paths, output_dir)


def main(args):
    import timeit
    start = timeit.default_timer()
    if args.pickle_file:
        feature_lst = pickle.load(open(args.pickle_file, 'rb'))
    else:
        features_to_run = [feat for feat in ALL_FEATURES if feat not in args.features_to_drop]
        feature_lst = FeatureList(args.hmmer_path, args.mmseqs_path, args.tmhmm_path, args.kmer,
                                  args.by_evalue, args.label_threshold, args.threshold_list, args.gene_window,
                                  args.nucleotide_window, features=features_to_run, n_cpus=args.ncpus)

    if args.dif_format_paths:
        create_features_pkls(feature_lst, args.input_path, args.dif_format_paths, args.output_dir)
    else:
        create_features_from_dir(args.input_path, feature_lst, args.suffix, args.output_dir)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("features Executed in ", execution_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run feature extraction on a given sample or a directory of samples.')
    parser.add_argument('--input_path', type=str, help='Insert the full path of the wanted directory with all the assemblies. not needed if --dif_format_paths is supplied.')
    parser.add_argument("--dif_format_paths", nargs='*', type=str,
                        help="The data in the 4 different formats faa,gff,ffn,fa. If we want to only run the script on one sample. if supplied --input_path is not needed.")
    parser.add_argument('--output_dir', type=str, help='the path to the directory we want to save our features in, default: "features" (new sub directory of current directory)', default='features')
    parser.add_argument('--hmmer_path', type=str, help="full path to the HMMER's hmmsearch program.")
    parser.add_argument('--mmseqs_path', type=str, help='full path to the Mmseqs2 program.')
    parser.add_argument('--tmhmm_path', type=str, help='full path to the tmhmm_path program.')
    parser.add_argument("-k", "--kmer", type=int, default=4, help="It will run the kmers count from 2 to k, default=4")
    parser.add_argument("-lt", "--label_threshold", type=str, default="1e-10",
                        help="Threshold for labeling of the data, default='1e-10'")
    parser.add_argument("-t", "--threshold_list", nargs='*', type=str, default=['1e-8'],
                        help="Thresholds for the proximity feature - hmm comparison, default=['1e-8']")
    parser.add_argument("-d", "--gene_window", type=int, default=10, help="Size of the ORFs window, default=10")
    parser.add_argument("-n", "--nucleotide_window", type=int, default=10000, help="Size of the nucleotides window, default=10000")
    parser.add_argument("--ncpus", type=int, default=64, help="Number of cpus to use for feature extraction, default=64")
    parser.add_argument("-e", "--by_evalue", action='store_true', dest='by_evalue',
                        help="choose this to use a threshold by e value (default). use --by_score for score threshold")
    parser.add_argument("-s", "--by_score", action='store_false', dest='by_evalue',
                        help="choose this to use a threshold by score. use --by_evalue for evalue threshold")
    parser.add_argument("-sf", "--suffix", default='.min10k.', help="suffix to sample files such that the protein file will end with {suffix}proteins.faa."
                                                                    " for example, .min10k. (default value) to get only contigs of length more than 10k. Input '' if none applies")

    parser.add_argument("-ftd", "--features_to_drop", nargs='*', type=str, default=['CARD_labeling', 'DNA_KMers', 'Cross_Membrane', 'GC_Content', 'SmartGC'],
                        help="The list of features names (according to the names in FeaturesList) that we dont want"
                             "to execute. default: ['CARD_labeling', 'DNA_KMers', 'Cross_Membrane', 'GC_Content', 'SmartGC']")
    parser.add_argument("-pkl", "--pickle_file", help="path to pickle file with a FeatureList object. default: '' "
                                                      "(no file, therefore a new object is created)",  type=str, default='')
    parser.set_defaults(by_evalue=True)
    args = parser.parse_args()
    main(args)
