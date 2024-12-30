import pandas as pd
from Bio import SeqIO
import re
import pickle
import mpmath as math
import subprocess
import glob
from abc import ABC, abstractmethod
import os
from functools import wraps
from pathlib import Path
from collections import Counter


MISSING_VALUE_SYMBOL = None
CONTIG_GENE_RATIO = 4
AMBIGUOUS_NUC = ["N", "M", "R", "Y", "W", "S", "X", "K", "B", "D", "H", "V", "Z"]
CONTIG_SIZE = '.min10k.'
SUFFIX_LIST = ['proteins.faa', 'gff', 'genes.ffn', 'fa']



class MLFeature(ABC):
    @abstractmethod
    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        pass


def getIDs(fastafile):
    '''
    :param fastafile:
    :return: a dataframe with just the names of the genes from the fasta file
    '''
    geneList = list(SeqIO.parse(fastafile, "fasta"))
    IDs = [record.id for record in geneList]
    return pd.DataFrame(IDs, columns=['ID'])


def getData(contigfasta, genesfasta):
    '''
    gets the fasta files of the contigs and the genes separately. than organize it into a dataframe when each row has a
    contig and a list of its genes.
    IMPORTANT: contigs with no genes are erased
    :param contigfasta: the contigs fasta file - The sequence of the entire Contig
    :param genesfasta: ORFs fasta file - Sequence of each gene is separate
    :return: a dataframe with the contig name, contig seq, and the the list of all the genes from that contig
    '''
    contiglist = list(SeqIO.parse(contigfasta, "fasta"))
    geneList = list(SeqIO.parse(genesfasta, "fasta"))
    all_contigs_and_genes = []
    gene_index = 0
    for contig in contiglist:
        ORFlist = []
        # finds the last '_' and cuts it out, so the name of the contig is left
        while gene_index < len(geneList) and geneList[gene_index].id[:geneList[gene_index].id.rfind("_")] == contig.id:
            ORFlist.append(geneList[gene_index])
            gene_index += 1
        if not ORFlist:
            continue
        all_contigs_and_genes.append([contig.id, contig.seq, ORFlist])
        if gene_index == len(geneList):
            break

    return pd.DataFrame(all_contigs_and_genes, columns=['contig_name', 'contig_seq', 'genes_list'])


def get_filtered_fasta(in_fasta, out_fastas, ids):
    records = filter_fasta(in_fasta, ids)
    SeqIO.write(records, out_fastas, "fasta")


def filter_fasta(in_fasta, ids):
    return [rec for rec in SeqIO.parse(in_fasta, "fasta") if rec.id in ids]


def create_fasta_from_df(df, source_path, output_location='relevant_seqs.fasta', suffix='.min10k.proteins.faa', is_contig=False):
    if os.path.exists(output_location):
        os.remove(output_location)
    inds = list(set(['_'.join(ind.split('_')[:-1]) for ind in df.index])) if is_contig else df.index.tolist()
    if os.path.isdir(source_path):  # given a directory with files instead of a single file
        rel_fastas = glob.glob(os.path.join(source_path, '**', f'*{suffix}'), recursive=True)
        recs = [filter_fasta(fasta, inds) for fasta in rel_fastas]
        SeqIO.write([item for sublist in recs for item in sublist], output_location, "fasta")
    else:
        get_filtered_fasta(source_path, output_location, inds)
    print("finished create_fasta_from_df")
    return output_location


def check_if_rel_fasta(sample_name, fasta):
    return any([sample_name in rec.id for rec in SeqIO.parse(fasta, 'fasta')])


def find_fasta_file(sample_name, source_path, suffix='.min10k.proteins.faa'):
    if isinstance(source_path, list): # input is a list of possible directories
        for dir_path in source_path:
            rel_fastas = glob.glob(os.path.join(dir_path, '**', f'*{suffix}'), recursive=True)
            for fasta in rel_fastas:
                if check_if_rel_fasta(sample_name, fasta):
                    return fasta

    elif os.path.isdir(source_path):  # input is a single directory
        rel_fastas = glob.glob(os.path.join(source_path, '**', f'*{suffix}'), recursive=True)
        for fasta in rel_fastas:
            if check_if_rel_fasta(sample_name, fasta):
                return fasta
    else:  # input is a single fasta file
        if check_if_rel_fasta(sample_name, source_path):
            return source_path
    return

def get_sample_name(sample_name):
    if '_ctg' in sample_name:
        file_to_find = sample_name.split('_ctg')[0]
    elif "_contig" in sample_name and "_bin" in sample_name:
        file_to_find = sample_name.split('_bin')[0]
    else:
        sample_parts = sample_name.split("|")[:3]
        file_to_find = "|".join(sample_parts)
    return file_to_find


def create_fasta_per_contig(df, output_dir, fastas_dir, suffix='.min10k.fa'):
    inds = df.index.tolist()
    contigs = set(["_".join(ind.split("_")[:-1]) for ind in inds])
    rel_fastas = glob.glob(os.path.join(fastas_dir, f'*{suffix}'))
    for contig in contigs:
        for fasta in rel_fastas:
            rel_recs = [rec for rec in SeqIO.parse(fasta, "fasta") if contig in rec.id]
            if rel_recs:  # found the relevant file with the contig
                SeqIO.write(rel_recs, os.path.join(output_dir, f"{contig.replace('|', '.')}.fasta"), "fasta")
                break


def combine_all_pkls(directory_path , df):
    '''
    runs over the directory and combines all the pkl into one.
    :param directory_path:
    :param df:
    :return:
    '''
    with os.scandir(directory_path) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                df = combine_all_pkls(entry.path ,df)  # Recursively run over the files
            else:
                if entry.name.endswith(".pkl") and entry.is_file():
                    # print(entry.name)
                    unpickled_df = pd.read_pickle(entry)
                    if unpickled_df.index.name != 'ID':
                        unpickled_df.set_index('ID', inplace=True)
                    df = df.join(unpickled_df, how='outer')
    # reduce memory
    floats = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
    f_dict = {col: 'float32' if "_distance_" in col or "_pp" in col else 'float16' for col in floats}
    f_dict = {k: v for k,v in f_dict.items() if k in df.columns}
    df = df.astype(f_dict)
    return df



def slidingwindow(seq, window, step):
    '''
    generic sliding window for sequences
    :param seq: the sequence
    :param window: window size
    :param step: this step size
    :return: yields a window at each iteration
    '''
    seqlen = len(seq)
    if seqlen < window:
        return None
    for i in range(0, seqlen, step):
        j = min(i + window, seqlen)
        yield seq[i:j]
        if j == seqlen:
            break


def k_Mers_Count(k, seq, step, is_dna=False):
    '''
    goes over the sequence using the window method, and return a dictionary of all the kmers and their count.
    :param k: int, the k of the k-mers, how big will the window be.
    :param seq: string, the letters to go over
    :param step: if we would like the window to skip some, like in codon usage instance step will be 3.
    :return: a dictionary with the kmers as keys and their count as value.
    '''
    mers_dict = Counter()
    for window in slidingwindow(seq, k, step):
        if (is_dna and any([nuc in window for nuc in AMBIGUOUS_NUC])) or len(window) < k:
            continue
        mers_dict[window.upper()] += 1
    return mers_dict


def create_relevant_directory(feature_directory, file_path, dir_path='features'):
    '''
    creates the directory to store all the future runs for this feature
    :param feature_directory: the name of the directory for the feature. it should be the name of the feature+dir
    :param file_path: path to sample file.
    :param dir_path: name of directory to save all the features
    '''
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    p = Path(Path(file_path).stem)
    sample_dir = p.stem if p.suffix =='.proteins' else p
    sample_dir = os.path.join(dir_path, sample_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(os.path.join(sample_dir, feature_directory)):
        os.makedirs(os.path.join(sample_dir, feature_directory))
    return os.path.join(sample_dir, feature_directory)


def check_file_exists(file_path, feature_name, directory, protocol='pkl'):
    '''
    checks whether the file exists.
    :param file_path: the path to the data file so we can extract the name we want to save the file by
    :param feature_name: the name of the feature
    :param directory: where the feature file should be saved
    :param protocol: the file protocol of the feature file.
    :return: 1 if file exists and has something stored.
    '''
    filename = os.path.join(directory, f'{str(Path(file_path).stem)}.{feature_name}.{protocol}')
    # file name might include '.proteins' and also might not
    filename = Path(filename) if Path(filename).is_file() else Path(filename.replace('.proteins', ''))
    if filename.is_file():
        filesize = os.stat(filename).st_size
        if filesize > 0:
            return 1
    return 0


def results_to_file(df, file_path, feature_name, directory):
    filename = os.path.join(directory, f'{str(Path(file_path).stem)}.{feature_name}.pkl')
    print('saving to ' + filename)
    print(df.shape)
    df.to_pickle(filename)


def results_to_csv(df, file_path, feature_name, directory):
    filename = os.path.join(directory, f'{str(Path(file_path).stem)}.{feature_name}.tsv')
    df.to_csv(filename, sep='\t', float_format='%.2f')


def feature_to_file(feature_name, protocol='pkl', dir_path='features'):
    '''
    a method that is used as a double decorator.
    the goal is the wrap a method that returns a dataframe, and save it into a file, if it doesn't exist, and name it.
    won't run the method if the file exists.

    :param feature_name: what should be the name of the file, ie GC_content
    :param protocol: the protocol to save into - pkl/csv
    :param dir_path: the path to the output_dir
    :return:
    '''
    def to_file(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if feature_name == 'ALL_PKLS':
                feature_dir = 'all_merged_pkls'
                if not os.path.exists(feature_dir):
                    os.makedirs(feature_dir)
                    print('created directory')
            else:
                feature_dir = feature_name + '_dir'
                feature_dir = create_relevant_directory(feature_dir, args[0], dir_path)
            if not check_file_exists(args[0], feature_name, feature_dir, protocol):
                df = func(*args, **kwargs)
                # optimize the size of the dataframe
                df = optimize_the_df(df)
                if protocol == 'pkl':
                    results_to_file(df, args[0], feature_name, feature_dir)
                elif protocol == 'csv':
                    results_to_csv(df, args[0], feature_name, feature_dir)
        return wrapper
    return to_file


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame: # https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_the_df(df):
    '''
    dataframes are usually not that great at optimazing themselfs, this soppose to make the size of the dataframe smaller.
    (based on this: https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
    for now missing a step.
    '''
    compact_df = optimize_floats(df)
    # compact_df = optimize_floats(compact_df)
    return compact_df


def extract_samples(dirpath, size=CONTIG_SIZE):
    """
    go through the directory and return the files of all the samples that have the 4 formats .gff, .fa, .faa, .fnn
    :param dirpath: a path to a directory that we know that contains files
    :return: a list of lists, containing the 4 formats of the samples
    """
    list_of_samples = []

    prot_files = glob.glob(os.path.join(dirpath, f'*{size}proteins.faa'))
    print(f'the prots:{prot_files}')
    for full_path_entity in prot_files:
        if os.path.getsize(full_path_entity) > 0:
            curr_dir, entity = os.path.split(full_path_entity)
            sample_general_name = str(entity).split(size+'proteins.faa')[0]
            # we assume all the relevant files are in the same directory
            four_paths = [os.path.join(curr_dir, f'{sample_general_name+size+ending}') for ending in SUFFIX_LIST]
            if all([os.path.exists(file_path) for file_path in four_paths]):  # make sure we found them all
                list_of_samples.append(four_paths)

    return list_of_samples


def go_through_files(dir_path, size=CONTIG_SIZE):
    """
    goes through all the sample files and run all the features on each sample
    :param args: the arguments recieved from the user
    :param dir_path: starts with the dir_path given by the user, keep entering the folders
    :return: for each sample, df wrapped as pkl file, in the folder "all_merged_pkls"
    """

    files_paths_list = extract_samples(dir_path, size)
    print(f"len is {len(files_paths_list)}")
    if not files_paths_list:  # if contains - we assume we no need to look at sub-folders
        for entity in os.scandir(dir_path):
            if entity.is_dir():
                entity = os.path.join(dir_path, entity)
                inner_list = go_through_files(entity, size)
                print(f"len is {len(inner_list)}")
                files_paths_list.extend(inner_list)
    return files_paths_list


def fill_all_empty_orfs_with_zeros(df, ids):
    df.set_index('ID', inplace=True)
    idslist = list(ids)
    return df.reindex(idslist, fill_value=0, copy=True)


def get_exponent(evalue):
    exponent = -math.log(evalue, 10)
    exponent = float(exponent) if exponent != math.inf else 1000.0
    return exponent


def parse_col_name(col):
    return col.replace("_DB_reducted", "").replace('Bacteria_', '').replace('_pp', '').replace('[', ' [').replace("_", " ").replace("all ARGs filtered", 'ARGs')


def parse_arg_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("'", '')


def parse_query_name(query_name):
    return re.sub('(_[0-9]+)$', '', query_name).replace("'", '') if pd.notnull(query_name) else query_name



def load_param_pkl(param_pkl):
    param_dict = None
    if param_pkl:
        with open(param_pkl, "rb") as f_in:
            param_dict = pickle.load(f_in)
            if "random_state" in param_dict:
                del param_dict['random_state']
    return param_dict