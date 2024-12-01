import pandas as pd
import os
import re
from Bio import SeqIO
import itertools
import sys
from run_mmseqs_search import run_mmseqs_on_dir_rec
from utilities import get_sample_name, find_fasta_file


COLS = ["result_id", "ID", "identity", "alignment_length", "mismatch_num", "gap_opening_num", "query_start", "query_end", "target_start", "target_end", "evalue", "bit_score"]
CLUSTER_SIZE_RATIO = 0.5


def get_gene_neighbors_id(seq_id, n=3):
    splited_id = seq_id.split("_")
    prefix, gene_id = "_".join(splited_id[:-1]), splited_id[-1]
    genes_to_add = [str(int(gene_id)+i).zfill(len(gene_id)) for i in range(1, n+1)]
    genes_to_add += [str(int(gene_id)-i).zfill(len(gene_id)) for i in range(1, n+1) if int(gene_id) > i]
    neighbors = [f"{prefix}_{gene}" for gene in genes_to_add]
    return sorted(neighbors)


def get_neighbors_fasta(df, source_path, output_location, n=3, suffix='.min10k.proteins.faa'):
    the_id_list = list(df.index)
    rel_recs = []
    if os.path.exists(output_location):  # no need to create the file
        return

    # groupby contig and get their names
    sample_groups = itertools.groupby(the_id_list, get_sample_name)
    for sample, group in sample_groups:
        if sample == 'ID':
            continue
        location = find_fasta_file(sample, source_path, suffix=suffix)
        if location is None:
            print(f"{location} location of sample {sample} does not exist. group is {list(group)}")
            continue
        records = SeqIO.parse(location, "fasta")
        wanted_ids = [get_gene_neighbors_id(ind, n) for ind in group]
        wanted_ids = set([item for sublist in wanted_ids for item in sublist])
        rel_recs += [rec for rec in records if rec.id in wanted_ids]
    SeqIO.write(rel_recs, output_location, "fasta")


def get_common_proteins(cluster_file, size_threshold):
    with open(cluster_file, "r") as f_in:
        text = f_in.read()
    clusters = re.split(">Cluster [0-9]+", text)[1:]
    clusters = [clu.split(">")[1:] for clu in clusters]
    clusters_dict = {list(filter(lambda x: '... *' in x, clu))[0].split("...")[0]: set(["_".join(ind.split("_")[:-1]) for ind in clu]) for clu in clusters}
    return [prot for prot, cluster in clusters_dict.items() if len(cluster) >= size_threshold]


def create_neighbor_clusters(df, fasta_ref_dirs, output_location, cluster_output_file, cdhit_path, retry=0):
    get_neighbors_fasta(df, fasta_ref_dirs, output_location)
    os.system(f"{cdhit_path} -T 6 -M 0 -g 1 -s 0.5 -c 0.7 -d 10000000 -i {output_location} -o {cluster_output_file}")

    if not os.path.exists(cluster_output_file + ".clstr") and retry < 3:
        create_neighbor_clusters(df, fasta_ref_dirs, output_location, cluster_output_file, cdhit_path, retry=retry+1)


def get_neighbors(info_df, fasta, ref_fasta_dirs, cdhit_path, mmseqs_path, mmseq_res_path, search_db, run_mmseqs=False, create_res_files=False):
    info_df = info_df.set_index("ID") if "ID" in info_df.columns else info_df.copy().iloc[:,:2]
    res_dir = os.path.join(mmseq_res_path, "res_by_id")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    if run_mmseqs:
        if not os.path.exists(search_db):
            os.system(command=f"""{mmseqs_path} createdb "{fasta}" "{search_db}" """)
        for dir_path in ref_fasta_dirs:
            run_mmseqs_on_dir_rec(dir_path, mmseq_res_path, search_db, mmseqs_path, 1e-10)

    if create_res_files:
        df_lst = []
        for file_name in os.listdir(mmseq_res_path):
            file_path = os.path.join(mmseq_res_path, file_name)
            if os.path.isdir(file_path) or "_mmseqs_res" not in file_name:
                continue
            df = pd.read_csv(file_path, sep="\t", names=COLS)[["ID", "result_id", "identity", "evalue"]]
            df_lst.append(df)

        df = pd.concat(df_lst)
        for ind, group in df.groupby('ID'):
            file_name = ind.replace("|", ".") + "_all_mmseqs_res.tsv"
            group.to_csv(os.path.join(res_dir, file_name), sep="\t", index=False)

    for file_name in os.listdir(res_dir):
        file_path = os.path.join(res_dir, file_name)
        if os.path.isdir(file_path) or ".tsv" not in file_path:
            continue
        print(file_name)
        df = pd.read_csv(file_path, sep="\t").set_index('result_id')
        gene_num = len(set(df.index))
        output_location = os.path.join(res_dir, file_name.replace("_all_mmseqs_res.tsv", "_neighbors.fasta"))
        cluster_output_file = os.path.join(res_dir, file_name.replace("_all_mmseqs_res.tsv", "_neighbors_clustered.fasta"))
        if not os.path.exists(cluster_output_file):
            create_neighbor_clusters(df, ref_fasta_dirs, output_location, cluster_output_file, cdhit_path)
        repeating_prots = get_common_proteins(cluster_output_file + ".clstr", gene_num * CLUSTER_SIZE_RATIO)
        records = [seq for seq in SeqIO.parse(cluster_output_file, "fasta") if seq.id in repeating_prots]
        gene_id = df['ID'].iloc[0]
        info_df.loc[gene_id, "frequent_neighbors"] = len(repeating_prots)
        if len(records) > 0:
            SeqIO.write(records, file_path.replace("_all_mmseqs_res.tsv", "_repeating_neighbors.fasta"), "fasta")
    return info_df["frequent_neighbors"].to_frame()
