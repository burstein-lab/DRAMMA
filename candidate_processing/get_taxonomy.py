import pandas as pd
import numpy as np
import os
import re
import subprocess
from utilities import get_filtered_fasta, create_fasta_per_contig
from candidate_annotation_filtration import run_blast
from pathlib import Path


TAX_ID_TO_DESC = pd.read_pickle(os.path.join(Path(__name__).parent.absolute(), 'data', 'WGS_genomes_lineage_mapping.pkl'))
WGS_TAX_MAPPING = pd.read_pickle(os.path.join(Path(__name__).parent.absolute(), 'data', 'WGS_gca_to_tax_mapping.pkl'))
MMSEQS_COLS = ['ID', 'tax_id', 'tax_level', 'tax_name']
TMP_DIR = os.path.join(os.getcwd(), 'tax_tmp')
CONTIG_DIR =  os.path.join(os.getcwd(), 'candidates_contigs')
MMSEQS_RES_DIR = os.path.join(CONTIG_DIR, "mmseq")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CONTIG_DIR, exist_ok=True)
os.makedirs(MMSEQS_RES_DIR, exist_ok=True)


def get_taxonomy_by_id(ind):
    if "metagenome" in ind.lower():
        return
    gca = re.search("GCA_[0-9]+\.[0-9]", ind)
    if gca is not None and gca.group() in WGS_TAX_MAPPING.index:
        tax_id = WGS_TAX_MAPPING.loc[gca.group(), 'taxid']
        lineage = TAX_ID_TO_DESC.loc[tax_id, 'lineage']
        if 'metagenome' not in lineage:
            return tax_id


def fill_missing_tax_level(row, tax_level):
    if f"tax_level_{tax_level}" not in row.index:
        return np.nan
    if pd.notnull(row[f"tax_level_{tax_level}"]):
        return row[f"tax_level_{tax_level}"]
    return fill_missing_tax_level(row, tax_level-1)


def tax_id_to_lineage(tax_id):
    if pd.isnull(tax_id):
        return []
    if tax_id in TAX_ID_TO_DESC.index:
        return TAX_ID_TO_DESC.loc[tax_id, 'lineage'].split("; ")
    candidates = TAX_ID_TO_DESC[TAX_ID_TO_DESC['species_taxid'] == tax_id]
    if len(candidates) > 0:
        return candidates.iloc[0]['lineage'].split("; ")
    return []


def get_tax_hierarchy(df, filter_euk=False):
    df['tax_lst'] = df['tax_id'].apply(tax_id_to_lineage)
    if filter_euk:
        df['domain'] = df['tax_lst'].apply(lambda lst: lst[1] if len(lst) > 0 else "")
        df = df[df['domain'] != 'Eukaryota'].drop(columns=['domain'])
    df['tax_lst'] = df['tax_lst'].apply(lambda lst: lst[2:] if len(lst) > 0 else lst)
    max_len = df['tax_lst'].apply(len).max()
    new_cols = [f"tax_level_{i + 1}" for i in range(max_len)]
    df[new_cols] = df['tax_lst'].apply(lambda lst: pd.Series(lst + [None] * (max_len - len(lst))))
    return df


def run_mmseqs_taxonomy(inds, mmseqs_db, mmseqs_path, threads=48, sensitivity=4):
    params = f"--threads {threads} -s {sensitivity}"
    file_paths = [os.path.join(CONTIG_DIR, "_".join(ind.replace("|", ".").split("_")[:-1])+".fasta") for ind in inds]
    for file_path in file_paths:
        query_base = os.path.basename(file_path).split(".f")[0]
        query_db = os.path.join(MMSEQS_RES_DIR, f"{query_base}.mmdb")
        tax_result = os.path.join(MMSEQS_RES_DIR, f"{query_base}_vs_{mmseqs_db.replace('._mmdb', '')}_TaxResult._mmdb")

        cmd = f"{mmseqs_path} createdb {file_path} {query_db} && " \
              f"{mmseqs_path} taxonomy {query_db} {mmseqs_db} {tax_result} {TMP_DIR} {params} && " \
              f"{mmseqs_path} createtsv {query_db} {tax_result} {tax_result.replace('._mmdb', '.tsv')} && " \
              f"{mmseqs_path} taxonomyreport {mmseqs_db} {tax_result} {tax_result.replace('._mmdb', '.report')}"

        subprocess.run([cmd], capture_output=True, text=True, shell=True)


def get_taxonomy_by_mmseq(df, fastas_dir, mmseqs_db, mmseqs_path, extract_contigs=True, run_mmseq=True, rerun=False, suffix='.min10k.fa'):
    indices = list(df[pd.isnull(df['tax_id'])].index)
    if extract_contigs:
        create_fasta_per_contig(df[pd.isnull(df['tax_id'])], CONTIG_DIR, fastas_dir, suffix=suffix)

    if run_mmseq:
        run_mmseqs_taxonomy(indices, mmseqs_db, mmseqs_path)

    for ind in indices:
        filename = "_".join(ind.replace("|", ".").split("_")[:-1])
        file_path = os.path.join(MMSEQS_RES_DIR, filename + "_TaxResult.tsv")
        if rerun and (not os.path.exists(file_path) or len(pd.read_csv(file_path, names=MMSEQS_COLS, sep='\t')) == 0):
            run_mmseqs_taxonomy([ind], mmseqs_db, mmseqs_path)
        if not os.path.exists(file_path):
            continue
        m_df = pd.read_csv(file_path, sep='\t', names=MMSEQS_COLS).drop_duplicates().set_index('ID')
        if len(m_df) == 0 or ind not in m_df.index:
            continue
        tax_id = m_df.loc[ind, 'tax_id']
        if tax_id != 0:
            df.loc[ind, 'tax_id'] = tax_id


def get_taxonomy_by_blast(df, inds, blast_fasta, blast_db, diamond_path, get_blast=True):
    out_file = os.path.join(os.getcwd(), "blast_tax_results.tsv")
    if get_blast:
        run_blast(blast_fasta, out_file, diamond_path, 1e-6, db=blast_db)
    bdf = pd.read_csv(out_file, sep='\t', names=["ID", "qtitle", "sseqid", "stitle", "evalue"]).drop_duplicates()
    bdf = bdf.sort_values('evalue').drop_duplicates(subset=['ID']).set_index('ID')  # taking only best result of each id
    bdf['tax_id'] = bdf['stitle'].apply(lambda x: int(re.search('(?<=TaxID\=)[0-9]+', x).group()) if re.search('(?<=TaxID\=)[0-9]+', x) is not None else None)
    inds = [ind for ind in inds if ind in bdf.index]
    df.loc[inds, 'tax_id'] = bdf.loc[inds, 'tax_id']


def get_taxonomy(df, fasta, fastas_dir, mmseqs_db, mmseqs_path, blast_db, diamond_path, extract_contigs=True, run_mmseq=True, suffix='.min10k.fa', to_delete=True, filter_euk=False):
    df = df.drop_duplicates() if "ID" in df.columns else df.reset_index().drop_duplicates()
    df['tax_id'] = df['ID'].apply(get_taxonomy_by_id)
    df.set_index('ID', inplace=True)
    get_taxonomy_by_mmseq(df, fastas_dir, mmseqs_db, mmseqs_path, extract_contigs, run_mmseq, suffix=suffix)
    unknown_ids = list(df[pd.isnull(df['tax_id'])].index)
    blast_fasta = os.path.join(os.getcwd(), "ids_for_tax_blast.fasta")
    get_filtered_fasta(fasta, blast_fasta, unknown_ids)
    get_taxonomy_by_blast(df, unknown_ids, blast_fasta, blast_db, diamond_path, get_blast=run_mmseq)
    df = get_tax_hierarchy(df, filter_euk=filter_euk)

    if to_delete:
        to_rm = [blast_fasta, TMP_DIR, CONTIG_DIR, os.path.join(os.getcwd(), "blast_tax_results.tsv")]
        for path in to_rm:
            os.system(f'rm -r {path}')
    return df