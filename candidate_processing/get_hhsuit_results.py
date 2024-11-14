import re
import numpy as np
import pandas as pd
from Bio import SeqIO
import time
import os
import subprocess


HEADER = " No Hit                             Prob E-value P-value  Score    SS Cols Query HMM  Template HMM"
TMP_DIR = os.path.join(os.getcwd(), "hhsuit_tmp")
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)


def parse_result(result, total_stats, i):
    lines = result.strip().split('\n')
    desc = lines[0].lstrip(">")
    eval = re.search("E\-value\=\S+", lines[1]).group().replace("E-value=", "")
    match_length = re.search("Aligned_cols\=[0-9]+", lines[1]).group().replace("Aligned_cols=", "")
    total_length = re.search("\([0-9]+\)", total_stats[i]).group().strip("()")
    match_fraction = int(match_length) / int(total_length) if int(total_length) != 0 else np.nan
    return {"description": desc, "e-value": eval, "hit_length_fraction": match_fraction}


def parse_hhsuit_results(res_file):
    with open(res_file, "r") as f_in:
        text = f_in.read()
    results = re.split("No [0-9]+", text)
    total_stats, results = results[0], results[1:]
    total_stats = total_stats.split("\n")
    index = total_stats.index(HEADER)
    total_stats = total_stats[index+1:]
    dict_lst = [parse_result(res, total_stats, i) for (i, res) in enumerate(results)]
    if not dict_lst:
        return
    df = pd.DataFrame(dict_lst)
    df['is_domain'] = df.loc[:, 'hit_length_fraction'] <= 0.4
    return df


def get_hhsuit_res_per_seq(res_prefix, seq_id):
    df_lst = []
    for db in ["pfam", "pdb"]:
        df = parse_hhsuit_results(f"{res_prefix}_{db}.hhr")
        if df is not None:
            df = df.reindex(columns=df.columns.tolist() + ["DB"], fill_value=db)
            df_lst.append(df)
    if not df_lst:
        return pd.DataFrame([]).rename_axis('ID')
    df = pd.concat(df_lst) if len(df_lst) > 1 else df_lst[0]
    cols = df.columns.tolist()
    df = df.reindex(columns=df.columns.tolist() + ['ID'], fill_value=seq_id)
    return df[['ID']+cols]


def run_hhsuit(seq, hhblits_path, pfam_db, pdb_db, e_val=1e-10, hit_num=500, ncpus=4):
    e_val = str(e_val).replace("e", "E")
    filename = os.path.join(TMP_DIR, f"tmp_seq_for_hhsuit_{time.time()}.fasta")
    res_name = filename.replace(".fasta", "")
    SeqIO.write(seq, filename, "fasta")
    command = f"""{hhblits_path} -cpu {ncpus} -i {filename} -d {pfam_db} -n 3 -E {e_val} -Z {hit_num} -B {hit_num} -o {res_name}_pfam.hhr &&""" \
    f"""{hhblits_path} -cpu {ncpus} -i {filename} -d {pdb_db} -n 3 -E {e_val} -Z {hit_num} -B {hit_num} -o {res_name}_pdb.hhr"""
    process_count = subprocess.run([command], capture_output=True, text=True, shell=True)
    return res_name, seq.id


def remove_files(res_prefix):
    for suffix in [".fasta","_pfam.hhr", "_pdb.hhr"]:
        os.remove(res_prefix+suffix)


def get_hhsuit_results(fasta_file, hhblits_path, pfam_db, pdb_db, e_val=1e-10, hit_num=500, ncpus=4):
    df_lst = []
    seq_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        res_file, seq_id = run_hhsuit(record, hhblits_path, pfam_db, pdb_db, e_val, hit_num, ncpus)
        df = get_hhsuit_res_per_seq(res_file, seq_id)
        remove_files(res_file)
        df_lst.append(df)
        seq_ids.append(seq_id)

    if df_lst:
        final_df = pd.concat(df_lst).set_index('ID')
        final_df['e-value'] = final_df['e-value'].astype('float64')
        final_df = final_df[final_df['e-value'] < e_val]  # filtering eval again as hhsuit can't be trusted
        IDS_df = pd.DataFrame(seq_ids, columns=['ID']).set_index('ID')
        final_df = IDS_df.join(final_df, how='outer')
    else:
        final_df = pd.DataFrame([])
    return final_df
