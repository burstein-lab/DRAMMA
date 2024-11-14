import os
import pickle
import numpy as np
import pandas as pd
import json
import sys
from Bio import SeqIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from utilities import get_filtered_fasta, create_fasta_from_df
from get_hhsuit_results import get_hhsuit_results
from feature_extraction.create_tblout_file import get_tblout_file
from feature_extraction.analyze_tblout_result import process_tblout_file
import matplotlib
import seaborn as sns
import matplotlib.colors as colors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess


ANNOTATION_MAPPING_FILE = os.path.join(Path(__name__).parent.absolute(), 'data', 'annotation_mapping.pkl')
with open(ANNOTATION_MAPPING_FILE, 'rb') as fin:
    ANNOTATION_MAPPING = pickle.load(fin)

KO_MAPPING_FILE = os.path.join(Path(__name__).parent.absolute(), 'data', 'KO_to_desc.json')
with open(KO_MAPPING_FILE, 'r') as fin:
    KO_DESC = json.load(fin)

UNKNOWN_PATTERN = "|".join(ANNOTATION_MAPPING['unknown_terms']).replace("(", "\(").replace(")", "\)")
DOMAIN_BINDING_PATTERN = "|".join(ANNOTATION_MAPPING['domain_binding_keywords']).replace("(", "\(").replace(")", "\)")
ARG_PATTERN = "|".join(ANNOTATION_MAPPING['ARG_keywords']).replace("(", "\(").replace(")", "\)")
ARG_KOS = ANNOTATION_MAPPING['ARG_KOs']
DESC_MAPPING = ANNOTATION_MAPPING['description_mapping']
KNOWN_CLASSES = ANNOTATION_MAPPING['known_classes']
ARG_CLASSES = ANNOTATION_MAPPING['ARG_classes']
TMP_DIR = os.path.join(os.getcwd(), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def filter_high_score_candidates(df, threshold):
    df = df[df['Label'] == 0]
    if f'passed_{round(threshold, 2)}' in df.columns:
        df = df[df[f'passed_{round(threshold, 2)}'] == True]
    return df.drop(columns=[col for col in df.columns if 'passed_' in col])


def get_class_percentages(df, unknowns=()):
    if len(unknowns) > 0:
        df = pd.concat([df,  pd.DataFrame(['Unknown']*len(unknowns), index=list(unknowns), columns=['Class'])])
    return df['Class'].value_counts(normalize=True).to_dict()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_classes_bar_plot(df, image_path):
    # Create a stacked bar plot
    color_map = sns.color_palette('Purples', len(df.columns) + 3, as_cmap=True)
    new_cmap = truncate_colormap(color_map, 0.25, 1)  # Making sure the first color is not too bright
    ax = df.plot(kind='bar', stacked=True, figsize=(20, 12), rot=0.3, colormap=new_cmap)

    # Set labels and title
    plt.xlabel('Pipeline step', fontsize=22, labelpad=22)
    plt.ylabel('Percentage (%)', fontsize=22)

    # Move the legend outside the plot and increase legend font size
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=20)

    # Increase tick label font size for both x and y axes
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Display values on top of the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if round(height*100, 1) > 0.0:
            x, y = p.get_xy()
            ax.annotate(f'{round(height*100, 1)}%', (x + width + 0.08, y + height/2), ha='center', fontsize=18)

    # Show the plot
    plt.savefig(image_path, bbox_inches='tight')


def get_KEGG_annotations(hmmer_path, hmm_file_name, fasta, tblout_path, threshold=1e-6, ncpus=8):
    get_tblout_file(hmmer_path, hmm_file_name, fasta, retry=3, is_domain=False, tblout_path=tblout_path, cpu=ncpus)
    hmm_df = process_tblout_file(tblout_path, threshold, by_e_value=True)  # turns it to DF
    hmm_df['KO'] = hmm_df['query_name'].apply(lambda x: x.split('.')[0] if pd.notnull(x) and len(x) > 0 else "-")
    hmm_df['desc'] = hmm_df['KO'].apply(lambda x: KO_DESC[x] if x in KO_DESC else 'hypothetical protein')
    return hmm_df[['ID', 'KO', 'desc']]


def map_KEGG_annotations(res_df, res_dict, fasta_source_path, hmmer_path, kegg_hmm, fasta_suffix, output_prefix, ncpus=8):
    print("in KEGG")
    KEGG_output_file = output_prefix + '_KEGG_results.pkl'
    fasta_for_KEGG = output_prefix + '_sequences_for_KEGG.fasta'
    if os.path.exists(KEGG_output_file):
        with open(KEGG_output_file, 'rb') as fin:
            objs = [pickle.load(fin) for i in range(5)]
            res_dict, res_df, ids_to_check, known, class_mapping = objs
        return res_dict, res_df, ids_to_check, known, class_mapping, fasta_for_KEGG

    if not os.path.exists(fasta_for_KEGG):
        create_fasta_from_df(res_df, fasta_source_path, output_location=fasta_for_KEGG, suffix=fasta_suffix)

    # hmmer_path, hmm_file_name, fasta, tblout_path, threshold=1e-6, ncpus=8
    tblout_path = output_prefix + '_KEGG_results.tblout'
    res_df = get_KEGG_annotations(hmmer_path, kegg_hmm, fasta_for_KEGG, tblout_path, ncpus)
    unknown_inds = res_df[(pd.notnull(res_df['desc'])) & (res_df['desc'].str.contains(UNKNOWN_PATTERN, case=False))].index
    domain_binding_inds = res_df[(pd.notnull(res_df['desc'])) & (res_df['desc'].str.contains(DOMAIN_BINDING_PATTERN, case=False))].index
    res_df['Class'] = res_df['desc'].map(DESC_MAPPING)

    # updating classes using terms not found in DESC_MAPPING
    res_df[['desc', 'KO']] = res_df[['desc', 'KO']].fillna('')
    res_df.loc[res_df['desc'].str.contains(ARG_PATTERN, case=False), 'Class'] = 'Antimicrobial Resistance Gene (ARG)'
    res_df.loc[res_df[res_df['KO'].isin(ARG_KOS)].index, 'Class'] = 'Antimicrobial Resistance Gene (ARG)'
    res_df.loc[unknown_inds, 'Class'] = 'Unknown'
    res_df.loc[domain_binding_inds, 'Class'] = 'Annotated (Not ARG)'
    res_df.set_index('ID', inplace=True)
    res_df['Class'] = res_df['Class'].fillna("Annotated (Not ARG)")

    class_mapping = get_class_percentages(res_df)
    res_df = res_df.reset_index() if "ID" not in res_df.columns else res_df
    ids_to_check = res_df[res_df['Class'] == 'Unknown'][['ID', "Class"]].drop_duplicates().set_index("ID")
    known = res_df[res_df['Class'].isin(KNOWN_CLASSES)]
    res_dict['KEGG_known'] = known
    res_dict['KEGG_ARG'] = res_df[res_df['Class'].isin(ARG_CLASSES)]

    with open(KEGG_output_file, 'wb') as fout:
        objs = [res_dict, res_df, ids_to_check, known, class_mapping]
        [pickle.dump(obj, fout) for obj in objs]

    return res_dict, res_df, ids_to_check, known, class_mapping, fasta_for_KEGG


def run_clustering(fasta, mmseqs_path, ncpus=8):
    print("in clustering")
    prefix = fasta.replace(".fasta", "")
    command = f"""{mmseqs_path} easy-cluster {fasta} {prefix} {TMP_DIR} -s 7.5 -c 0.5 --threads {ncpus} -v 2"""
    process_count = subprocess.run([command], capture_output=True, text=True, shell=True)
    return prefix + "_rep_seq.fasta"


def run_blast(fasta, out_file, diamond_path, eval, db, ncpus=8, id_thresh=None):
    id_str = "" if id_thresh is None else f"--id {id_thresh} "
    command = f"""{diamond_path} blastp -d {db} -q {fasta} -e {eval} {id_str}--threads {ncpus} -v --sensitive --tmpdir {TMP_DIR} -o {out_file} -f 6 qseqid qtitle sseqid stitle evalue"""
    process_count = subprocess.run([command], capture_output=True, text=True, shell=True)


def get_blast_results(all_ids, res_dict, fasta, output_prefix, eval, diamond_path, blast_db, ncpus=8):
    print("in BLAST")
    blast_output_file = output_prefix + '_BLAST_results.pkl'
    blast_res_file = output_prefix + "_BLAST_results.tsv"

    if os.path.exists(blast_output_file):
        with open(blast_output_file, "rb") as fin:
            objs = [pickle.load(fin) for ob in range(4)]
            res_dict, unknowns, bdf, class_mapping = objs
            return res_dict, unknowns, bdf, class_mapping

    if not os.path.exists(blast_res_file):
        run_blast(fasta, blast_res_file, diamond_path, eval, blast_db, ncpus=ncpus)
    bdf = pd.read_csv(blast_res_file, sep='\t', names=["ID", "qtitle", "sseqid", "stitle", "evalue"]).drop_duplicates()
    unknowns = set([q_id for q_id in all_ids if q_id not in list(bdf['ID'])])  # IDs not found in blast

    n_bdf = bdf[(~bdf['stitle'].str.contains(UNKNOWN_PATTERN, case=False))]  # removing results with unknown terms
    unknowns.update([q_id for q_id in set(bdf['ID']) if q_id not in list(n_bdf['ID'])])  # IDs we removed when filtering unknown terms

    n_bdf = n_bdf.sort_values('evalue', ascending=True).drop_duplicates(['ID'])  # keeping only result with best eval
    n_bdf['stitle'] = n_bdf.apply(lambda r: r['stitle'].replace(r["sseqid"], "").split("[")[0].replace('MULTISPECIES: ', '').strip(), axis=1)  # cleaning title
    n_bdf['Class'] = n_bdf['stitle'].map(DESC_MAPPING)
    n_bdf.loc[n_bdf['stitle'].str.contains(ARG_PATTERN, case=False), 'Class'] = 'Antimicrobial Resistance Gene (ARG)'
    n_bdf.reset_index(inplace=True)
    n_bdf.loc[n_bdf[n_bdf['stitle'].str.contains(DOMAIN_BINDING_PATTERN, case=False)].index, "Class"] = 'Annotated (Not ARG)'
    n_bdf.set_index('ID', inplace=True)
    n_bdf['Class'] = n_bdf['Class'].fillna("Annotated (Not ARG)")

    class_mapping = get_class_percentages(n_bdf, unknowns)
    res_dict['Blast_known'] = n_bdf[n_bdf['Class'].isin(KNOWN_CLASSES)]
    res_dict['Blast_ARG'] = n_bdf[n_bdf['Class'].isin(ARG_CLASSES)]

    with open(blast_output_file, 'wb') as fout:
        objs = [res_dict, unknowns, bdf, class_mapping]
        [pickle.dump(obj, fout) for obj in objs]

    return res_dict, unknowns, bdf, class_mapping


def get_hhsuit_class(row):
    if row["Domain-Binding"]:
        return "Annotated (Not ARG)"
    if row["Antimicrobial Resistance Gene (ARG)"]:
        return 'Antimicrobial Resistance Gene (ARG)'
    if row["Unknown"] or pd.isnull(row['description']):
        return "Unknown"
    for key, value in DESC_MAPPING.items():
        if pd.isnull(key) or key.strip() == '':
            continue
        elif key.lower() in row['description'].lower():
            if value == "Unknown":  # We assume that we already filtered out all unknowns
                continue
            return value
    return "Annotated (Not ARG)"


def filter_arg_hhsuit_results(df, output_path, ids_to_check):
    all_ids = list(df.index)
    unknown_ids = set([ind for ind in ids_to_check if ind not in all_ids])  # ids that did not yield hhsuit results
    df = df[pd.notnull(df['description'])].reset_index()
    unknown_inds = df[df['description'].str.contains(UNKNOWN_PATTERN, case=False)].index
    domain_binding_inds = df[df['description'].str.contains(DOMAIN_BINDING_PATTERN, case=False)].index
    df.loc[unknown_inds, "Unknown"] = True
    df.loc[domain_binding_inds, "Domain-Binding"] = True
    df.set_index('ID', inplace=True)
    arg_ids = set(df[df['description'].str.contains(ARG_PATTERN, case=False)].index)
    df.loc[arg_ids, "Antimicrobial Resistance Gene (ARG)"] = True
    df.loc[:, ["Antimicrobial Resistance Gene (ARG)", "Unknown", "Domain-Binding"]] = df.loc[:, ["Antimicrobial Resistance Gene (ARG)", "Unknown", "Domain-Binding"]].fillna(False)
    df.to_csv(output_path.replace(".tsv", "_unfiltered.tsv"), sep='\t')

    df = df[~df['Unknown']]  # removing unknown terms
    unknown_ids.update([seq_id for seq_id in all_ids if seq_id not in df.index])

    map_df = df.sort_values(['is_domain', 'e-value'], ascending=[True, True]).reset_index().drop_duplicates(['ID'])
    map_df['Class'] = map_df.apply(get_hhsuit_class, axis=1)
    class_mapping = get_class_percentages(map_df, unknown_ids)
    df = df[~df["Domain-Binding"]]  # removing domain-binding terms
    arg = df.loc[df.index.isin(arg_ids)].drop(columns=["Antimicrobial Resistance Gene (ARG)", "Unknown", "Domain-Binding"])
    final_df = df.loc[~df.index.isin(arg_ids)].drop(columns=["Antimicrobial Resistance Gene (ARG)", "Unknown", "Domain-Binding"])  # removing ARGs
    final_df.to_csv(output_path, sep='\t')
    return final_df, arg, unknown_ids, class_mapping


def get_hhsuit_mapping(unknown, res_dict, fasta, output_prefix, hhblits_path, pfam_db, pdb_db, ncpus=4):
    fasta_for_hhsuit = output_prefix + '_sequences_for_HH_suit.fasta'
    hhsuit_res_path = output_prefix + "_HH-suit_results.tsv"
    if not os.path.exists(fasta_for_hhsuit):
        get_filtered_fasta(fasta, fasta_for_hhsuit, unknown)

    if os.path.exists(output_prefix + '_full_hhsuit_res.pkl'):
        full_hhsuit_df = pd.read_pickle(output_prefix + '_full_hhsuit_res.pkl')
    else:
        full_hhsuit_df = get_hhsuit_results(fasta_for_hhsuit, hhblits_path, pfam_db, pdb_db, e_val=0.01, ncpus=ncpus)
        full_hhsuit_df.to_pickle(output_prefix + '_full_hhsuit_res.pkl')

    if len(full_hhsuit_df) > 0:
        hhsuit_df_filtered = full_hhsuit_df[full_hhsuit_df['e-value'] <= 1e-10]
        hhsuit_df, arg, unknown_ids, hhsuit_classes = filter_arg_hhsuit_results(hhsuit_df_filtered, hhsuit_res_path, unknown)
    else:
        hhsuit_df, arg, unknown_ids, hhsuit_classes = full_hhsuit_df, pd.DataFrame([]), [], {}

    res_dict['hhsuit'] = hhsuit_df
    res_dict['hhsuit_ARG'] = arg
    return hhsuit_df, full_hhsuit_df, unknown_ids, hhsuit_classes, fasta_for_hhsuit


def parse_blast_res(ids, blast_res):
    blast_df = blast_res.drop(columns=["qtitle"]).set_index("ID")
    ids_to_check = [sid for sid in ids if sid in blast_df.index]
    blast_df = blast_df.loc[ids_to_check, :]
    blast_df = blast_df.sort_values('evalue', ascending=True).reset_index().drop_duplicates(["ID"]).set_index('ID')
    blast_df['blast_res_title'] = blast_df.apply(lambda r: r['stitle'].replace(r["sseqid"], "").split("[")[0].strip(), axis=1)
    return blast_df


def unite_all_data(ids, original_df, blast_res, hhsuit_res):
    probs_df = original_df.loc[ids,:]
    blast_df = parse_blast_res(ids, blast_res)
    hhsuit_res = hhsuit_res.loc[[i for i in ids if i in hhsuit_res.index], :]
    hhsuit_res = hhsuit_res.sort_values('e-value', ascending=True).reset_index().drop_duplicates(['ID']).set_index('ID')
    res_df = pd.concat([probs_df, blast_df, hhsuit_res], axis=1)
    return res_df


def unite_candidates(original_df, unknowns, kegg_known_df, full_blast_df, blast_unknowns, hhsuit_res, ids_with_blast):
    # Taking top 200 prots with kegg annotation, top 100 without kegg (after clustering) and all blast unknowns
    kegg_knowns = kegg_known_df.sort_values('prob', ascending=False).iloc[:200]
    kegg_knowns['status'] = 'Kegg Known'
    kegg_unknowns = unite_all_data([ind for ind in ids_with_blast if ind not in blast_unknowns], original_df, full_blast_df, hhsuit_res).sort_values('prob', ascending=False).iloc[:100]
    kegg_unknowns['status'] = 'Kegg Unknown'
    blast_unknowns = unite_all_data([ind for ind in blast_unknowns if ind not in unknowns], original_df, full_blast_df, hhsuit_res)
    blast_unknowns['status'] = 'Blast Unknown'
    unknowns = unite_all_data(unknowns, original_df, full_blast_df, hhsuit_res)
    unknowns['status'] = 'HH-suit Unknown'
    df_lst = [df.set_index('ID') if 'ID' in df.columns else df for df in [kegg_knowns, kegg_unknowns, blast_unknowns, unknowns]]
    candidates = pd.concat(df_lst)
    return candidates


def filter_candidates(args):
    res_dict = {}
    output_prefix = args.output_prefix.rstrip('_')
    res_df = pd.read_pickle(args.input_file)
    res_df = filter_high_score_candidates(res_df, args.threshold)
    res_dict, res_df, keg_unknowns, kegg_known_df, kegg_classes, fasta_for_KEGG = map_KEGG_annotations(res_df, res_dict, args.fasta_source, args.hmmer_path, args.kegg_hmm, args.fasta_suffix, output_prefix, args.ncpus)

    fasta_for_blast = output_prefix + '_sequences_for_BLAST.fasta'
    clustered_fasta = fasta_for_blast.replace(".fasta", "_rep_seq.fasta")
    if not os.path.exists(fasta_for_blast):
        get_filtered_fasta(fasta_for_KEGG, fasta_for_blast, keg_unknowns)

    if not os.path.exists(clustered_fasta):
        clustered_fasta = run_clustering(fasta_for_blast, args.mmseqs_path, args.ncpus)  # clustering to remove similar proteins

    ids_to_check = [rec.id for rec in SeqIO.parse(clustered_fasta, "fasta")]  # only moving forward with ids in fasta
    res_dict, blast_unknowns, full_blast_df, blast_classes = get_blast_results(ids_to_check, res_dict, clustered_fasta, output_prefix, 1e-6, args.diamond_path, args.diamond_db, ncpus=args.ncpus)

    hhsuit_df, full_hhsuit_df, unknown_ids, hhsuit_classes, fasta_for_hhsuit = get_hhsuit_mapping(blast_unknowns, res_dict, clustered_fasta, output_prefix, args.hhblits_path, args.pfam_db, args.pdb_db, args.ncpus)

    # Plotting the annotation mapping of the three steps
    classes_mapping = {'KEGG': kegg_classes, 'BLAST': blast_classes, 'HH-suit': hhsuit_classes}
    res_dict['classes_mapping'] = classes_mapping
    class_df = pd.DataFrame.from_dict(classes_mapping).fillna(0.0).T
    image_path = output_prefix + "_candidates_annotation_mapping.pdf"
    plot_classes_bar_plot(class_df, image_path)

    unknowns = list(set(unknown_ids))
    res_dict['unknown'] = unknowns
    candidates_fasta = output_prefix + "_unknown_candidates_seqs.fasta"
    get_filtered_fasta(fasta_for_hhsuit, candidates_fasta, unknowns)
    print(f"unknowns are: {unknowns}")
    if unknowns:
        candidates = unite_candidates(res_df, unknowns, kegg_known_df, full_blast_df, blast_unknowns, full_hhsuit_df, ids_to_check)
        res_dict['candidate'] = candidates
        create_fasta_from_df(candidates, args.fasta_source, output_prefix + 'all_candidates_seqs.fasta', suffix=args.fasta_suffix)

    if args.remove_files:
        for tmp_file in [fasta_for_KEGG, fasta_for_blast, clustered_fasta, output_prefix + '_KEGG_results.pkl', output_prefix + '_KEGG_results.tblout', output_prefix + '_BLAST_results.pkl', output_prefix + "_BLAST_results.tsv", output_prefix + '_sequences_for_HH_suit.fasta', output_prefix + "_HH-suit_results.tsv", output_prefix + "_HH-suit_results_unfiltered.tsv", output_prefix + '_full_hhsuit_res.pkl']:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filter out proteins with known annotation')
    parser.add_argument("-in", "--input_file", type=str, help="path to model result pickle file.")
    parser.add_argument("-fs", "--fasta_source", type=str, help="path to either fasta file with the relevant candidate protein sequences or directory with the all fastas of the relevant candidate protein sequences.")
    parser.add_argument("-s", "--fasta_suffix", type=str, help="Suffix to the relevant protein fasta files. Only used if --fasta_source is a directory. default: '.proteins.faa'", default='.proteins.faa')
    parser.add_argument("-out", "--output_prefix", type=str, help="Prefix for output file and tmp files.")
    parser.add_argument("-t", "--threshold", type=float, default=0.75, help="The precision threshold according to which we filter the proteins. default: 0.75")
    parser.add_argument("--ncpus", type=int, default=8, help="Number of cpus to use for the external programs such as DIAMOND. default: 8")
    parser.add_argument('--hmmer_path', type=str, help='full path to the hmmer program.')
    parser.add_argument('--kegg_hmm', type=str, help='full path to the KEGG HMM DB you want to search against.')
    parser.add_argument('--mmseqs_path', type=str, help='full path to the mmseqs program.')
    parser.add_argument('--diamond_path', type=str, help='full path to the DIAMOND program.')
    parser.add_argument('--diamond_db', type=str, help='full path to the protein DIAMOND DB you want to search against.')
    parser.add_argument('--hhblits_path', type=str, help="full path to the HH-Suit's hhblits program")
    parser.add_argument('--pfam_db', type=str, help="full path to the HH-suit's Pfam DB you want to search against.")
    parser.add_argument('--pdb_db', type=str, help="full path to the HH-suit's PDB DB you want to search against.")
    parser.add_argument("-rf", '--remove_files', dest='remove_files', action='store_true', help='Choose this to remove intermidate files. default: False (keep_files)')
    parser.add_argument("-kf", '--keep_files', dest='remove_files', action='store_false', help='Choose this to keep intermidate files. default: True (keep_files)')
    parser.set_defaults(remove_files=False)
    args = parser.parse_args()
    filter_candidates(args)