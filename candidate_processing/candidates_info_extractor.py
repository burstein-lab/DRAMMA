import pandas as pd
from Bio import SeqIO
import os
import subprocess
import sys
import argparse
from pathlib import Path
import pickle
import networkx as nx
from candidate_annotation_filtration import run_clustering
from get_taxonomy import get_taxonomy, fill_missing_tax_level
from get_candidates_neighbors import get_neighbors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_model import get_model_results_multi_class
from feature_extraction.HTH_domain import HTHDomainFeatures
from feature_extraction.cross_membrane import CrossMembraneFeatures
from utilities import getIDs
from feature_extraction.train_dataset_creator import create_dataset_by_ids

DATA_PATH = os.path.join(Path(__name__).parent.absolute(), 'data', 'feature_extraction')
HTH_HMM = os.path.join(DATA_PATH, 'Pfam_HTH_domains.hmm')
TMP_DIR = os.path.join(os.getcwd(), 'tmp')
SEARCH_DB = os.path.join(TMP_DIR, 'candidate_ARGs_mmseqs_db')
os.makedirs(TMP_DIR, exist_ok=True)


RENAME_DICT = {"sseqid": "blast_res_id", "blast_res_title": "blast_res_description", "evalue": "blast_evalue",
                 "description": "hhpred_res_description", "e-value": "hhsuit_e-value",
                 "hit_length_fraction": "hhsuit_hit_length_fraction", 'is_domain': 'is_domain_(fraction_less_than_0.4)',
                 'DB': 'hhsuit_match_DB', 'Mechanism Classification_class': 'Mechanism Classification_res'}
COLS_TO_REMOVE = ['KO', 'desc', 'stitle', 'Mechanism Classification_prob'] + [f'tax_level_{i}' for i in range(5, 10)]
TAX_BLACKLIST = ['Firmicutes', 'Actinobacteria']


def get_HTH_domain(fasta_file, ids, hmmer_path):
    HTH = HTHDomainFeatures(hmmer_path, HTH_HMM)
    df = HTH.get_features(fasta_file, ids).reset_index().drop_duplicates()
    df['HTH'] = df['HTH_rep'] > 0
    return df[['ID', 'HTH']].set_index('ID')


def get_cross_membrane(fasta_file, tmhmm_path):
    cm = CrossMembraneFeatures(tmhmm_path)
    cm_df = cm.get_features(fasta_file)
    return cm_df


def get_sequence(ids, fasta):
    fasta_dict_lst = [{'ID': record.id, 'seq_len': len(str(record.seq)), "seq": str(record.seq)} for record in SeqIO.parse(fasta, "fasta") if record.id in ids]
    seq_df = pd.DataFrame(fasta_dict_lst).set_index('ID')
    return seq_df


def merge_clusters(clu_dict):
    # Creating a Graph according to the clusters and searching for connected components
    G = nx.from_dict_of_lists(clu_dict)
    groups = nx.connected_components(G)
    clu_lst = [sorted(list(group)) for group in groups]
    return clu_lst


def get_clusters(fasta, mmseqs_path, diamond_path, ncpus=8):
    output_file = os.path.join(os.getcwd(), "candidates_rep_vs_rep_blast_res.tsv")
    rep_fasta = run_clustering(fasta, mmseqs_path, ncpus=ncpus)
    cluster_df = pd.read_csv(rep_fasta.replace("_rep_seq.fasta", "_cluster.tsv"), sep="\t", names=['ID2', 'ID']) # TODO check if this file is returned

    command = f"""{diamond_path} blastp -d {rep_fasta} -q {rep_fasta} -e 1e-6 --threads {ncpus} --tmpdir {TMP_DIR} -o {output_file} -f 6 qseqid sseqid pident"""
    os.system(command)

    blast_df = pd.read_csv(output_file, sep='\t', names=['ID', 'ID2', "identity"]).drop('identity', axis=1)
    blast_df = pd.concat([blast_df, cluster_df], ignore_index=True).drop_duplicates()
    cluster_dict = blast_df.reset_index().set_index('ID').to_dict()['index']
    blast_df['cluster_1'] = blast_df['ID'].map(cluster_dict)
    blast_df['cluster_2'] = blast_df['ID2'].map(cluster_dict)
    clu_dict = {}
    for rep, df in blast_df.groupby('cluster_1'):
        clu_dict[rep] = list(set(df["cluster_2"]))
    clu_lst = merge_clusters(clu_dict)
    cluster_mapping = {c: min(clu) for clu in clu_lst for c in clu}
    blast_df['cluster_after_blast'] = blast_df['cluster_1'].map(cluster_mapping)
    return blast_df[['ID', 'cluster_after_blast']].drop_duplicates().set_index('ID')


def get_mechanisms(ids, mech_model, feature_dirs):
    features_file = os.path.join(os.getcwd(), "candidates_all_features.pkl")
    create_dataset_by_ids(ids, feature_dirs, features_file)
    multi_class_df = get_model_results_multi_class(mech_model, features_file)
    return multi_class_df


def filter_candidates(df, top_candidates=10):
    df = df.set_index('ID').sort_values('prob', ascending=False) if 'ID' in df.columns else df.sort_values('prob', ascending=False)
    df = df[df['cross_membrane_count'] == 0].drop('cross_membrane_count', axis=1)
    df = df[df['HTH'] == False].drop('HTH', axis=1)
    df = df[pd.notnull(df['tax_level_2'])]
    df = df[~df['tax_level_2'].isin(TAX_BLACKLIST)]
    df = df[df['frequent_neighbors'] == 0]
    df = df.drop_duplicates('cluster_after_blast', keep='first')
    df = df[['prob', 'Class', 'blast_res_description', 'blast_evalue', 'hhsuit_res_description', 'hhsuit_e-value', 'hhsuit_hit_length_fraction', 'Updated Resistance Mechanism_class', 'Updated Resistance Mechanism_prob', 'tax_level_1', 'tax_level_2', 'tax_level_3', 'tax_level_4']]
    top = df.iloc[:top_candidates]
    return top


def prepare_file(df):
    columns = list(df.columns)
    df = df[columns[:2] + columns[-2:] + columns[2:-2]]
    df['KO'] = df['KO'].fillna('-')
    df['KEGG_description'] = df['KEGG_description'].fillna('hypothetical protein')
    df['prob'] = df['prob'].round(3)
    df['Updated Resistance Mechanism_prob'] = df['Updated Resistance Mechanism_prob'].round(3)
    df = df.rename(columns={'prob': 'Model Score', 'Updated Resistance Mechanism_prob': 'Resistance Mechanism Score', 'Updated Resistance Mechanism_class': 'Predicted Resistance Mechanism'})
    df = df.loc[:, df.count() != 0]
    df['taxonomic_group'] = df.apply(lambda row: fill_missing_tax_level(row, 4), axis=1)
    df = df[['Model Score', 'Class', 'KEGG_description', 'blast_res_description', 'Predicted Resistance Mechanism', 'taxonomic_group']]
    df = df.rename(columns={col: col.replace("_", ' ').title().replace('Kegg', 'KEGG').replace('Res ', 'Result ') for col in df.columns})
    return df


def get_info_df(candidate_pkl):
    with open(candidate_pkl, 'rb') as fin:
        data = pickle.load(fin)['candidate']
    data = data.rename(columns={'desc': 'KEGG_description', 'description': 'hhsuit_res_description'})
    return data.sort_values('prob', ascending=False).reset_index().drop_duplicates(subset=['ID']).set_index('ID')


def main(candidate_pkl, seq_fasta, feature_dirs, fastas_dir, ref_fasta_dirs, mmseqs_dir, mech_model, mmseqs_path,
         diamond_path, hmmer_path, tmhmm_path, cdhit_path, mmseqs_db, blast_db, suffix='.min10k.fa',
         extract_contigs=False, run_mmseq=False, top_candidates=10):
    info_df = get_info_df(candidate_pkl)
    ids = getIDs(seq_fasta)
    seq_df = get_sequence(info_df.index, seq_fasta)
    cluster_df = get_clusters(seq_fasta, mmseqs_path, diamond_path)
    HTH_df = get_HTH_domain(seq_fasta, ids, hmmer_path)
    cm_df = get_cross_membrane(seq_fasta, tmhmm_path)
    mech_df = get_mechanisms(info_df.index, mech_model, feature_dirs)
    tax_df = get_taxonomy(info_df, seq_fasta, fastas_dir, mmseqs_db, mmseqs_path, blast_db, diamond_path, suffix=suffix, extract_contigs=extract_contigs, run_mmseq=run_mmseq).drop(info_df.columns, axis=1, errors='ignore')

    info_df = info_df.join([seq_df, cluster_df, HTH_df, mech_df, tax_df, cm_df]).drop(columns=COLS_TO_REMOVE, errors='ignore')
    info_df = info_df.rename(columns={k: v for k, v in RENAME_DICT.items() if k in info_df.columns})
    info_df[['hhsuit_res_id', 'hhsuit_res_description']] = info_df['hhsuit_res_description'].apply(lambda x: pd.Series([x, x]) if pd.isnull(x) else pd.Series([x.split(" ;")[0], " ;".join(x.split(" ;")[1:])]))
    info_df.to_pickle('candidates_enrichment_no_neighbors.pkl')
    neighbors_df = get_neighbors(info_df, seq_fasta, ref_fasta_dirs, cdhit_path, mmseqs_path, run_mmseqs=run_mmseq, create_res_files=run_mmseq, mmseq_res_path=mmseqs_dir, search_db=SEARCH_DB)

    info_df = info_df.join(neighbors_df)
    info_df.to_csv(candidate_pkl.replace(".pkl", "_with_additional_info.tsv"), sep='\t')

    filtered_candidates = filter_candidates(info_df, top_candidates)
    clean_df = prepare_file(filtered_candidates)
    clean_df.to_csv(candidate_pkl.replace(".pkl", f"_top_{top_candidates}_with_additional_info.tsv"), sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract more information about the given candidates, should be run after candidate_annotation_filtration')
    parser.add_argument("--pkl", type=str, help="path to pkl file with the dictionary with all the candidate info from candidate_annotation_filteration")
    parser.add_argument("-fa", "--fasta", type=str, help="path to fasta file of the candidates. Can use all_candidates_seqs.fasta created by candidate_annotation_filteration.")
    parser.add_argument("-fds", "--feature_dirs", nargs='*', type=str, help="Paths to directories with the extracted features created using run_features or run_DRAMMA_pipeline")
    parser.add_argument("--fastas_dir", type=str, help="path to directory with all the original fasta files of the candidates.")
    parser.add_argument("-refs", "--ref_fasta_dirs", nargs='*', type=str, help="Paths to directories with the all fastas we want to search our candidates against to detect common neighbors.")
    parser.add_argument("-md", "--mmseq_dir", type=str, help="path to dir to save mmseqs tax results")
    parser.add_argument("--mech_model", default=os.path.join(Path(__name__).parent.absolute(), "data", "models", "AMR_mechanism_multiclass_model.pkl"), type=str, help="path to pickle with the mechanism model, relevant cols and thresholds dict. default: ../data/models/AMR_mechanism_multiclass_model.pkl")
    parser.add_argument('--mmseqs_path', type=str, help='full path to the Mmseqs2 program.')
    parser.add_argument('--diamond_path', type=str, help='full path to the DIAMOND program.')
    parser.add_argument('--hmmer_path', type=str, help="full path to the HMMER's hmmsearch program.")
    parser.add_argument('--tmhmm_path', type=str, help='full path to the tmhmm program.')
    parser.add_argument('--cdhit_path', type=str, help='full path to the CD-HIT program.')
    parser.add_argument('--mmseqs_db', type=str, help="Path to the mmseqs database to search against to find the candidates' taxonomy")
    parser.add_argument('--diamond_db', type=str, help="Path to the DIAMOND database to search against to find the candidates' taxonomy")
    parser.add_argument("-sf", "--suffix", default='.min10k.fa', help="suffix to contig files. default: .min10k.fa")
    parser.add_argument("-ec", "--extract_contigs", action='store_true', dest='extract_contigs',
                        help="choose this to create a fasta per contig of the candidates (default). Should be used if this is the first time running this script for the given candidates. Use --reuse_contigs if these files were already created.")
    parser.add_argument("-re", "--reuse_contigs", action='store_false', dest='extract_contigs',
                        help="choose this to use existing contig files created in previous run.")
    parser.add_argument("-rmm", "--run_mmseq", action='store_true', dest='run_mmseq',
                        help="choose this to run mmseqs for taxonomy and common neighbors detection (default). Should be used if this is the first time running this script for the given candidates. Use --reuse_mmseq if these files were already created.")
    parser.add_argument("-remm", "--reuse_mmseq", action='store_false', dest='run_mmseq',
                        help="choose this to use mmseqs result files created in previous run.")
    parser.add_argument("-tc", "--top_candidates", type=int, default=10, help="Number of candidates to return, default: 10")
    parser.set_defaults(extract_contigs=True)
    parser.set_defaults(run_mmseq=True)
    args = parser.parse_args()
    main(args.pkl, args.fasta, args.feature_dirs, args.fastas_dir, args.ref_fasta_dirs, args.mmseq_dir, args.mech_model,
         args.mmseqs_path, args.diamond_path, args.hmmer_path, args.tmhmm_path, args.cdhit_path, args.mmseqs_db,
         args.diamond_db, args.suffix, args.extract_contigs, args.run_mmseq, top_candidates=args.top_candidates)
