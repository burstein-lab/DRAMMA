import argparse
import sys
import os
import time
from utilities import go_through_files
from pathlib import Path


TMP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'feature_extraction', 'data_for_tax_features', 'query_dbs', 'tmp')
NCPUS = 64
GMEM = 32
RES_COLUMNS = ['ID', 'target_id', 'target_header', 'e-value']
CONTIG_SIZE = '.min10k.'


def send_mmseqs_job(fasta, db_path, output_file, mmseqs_path, high_eval=1e-10, ncpus=NCPUS, gmem=GMEM):
    mmseq_output = output_file.replace(".tsv", "")
    tmp_dir = os.path.join(TMP_DIR, str(time.time()))
    command = f"""{mmseqs_path} easy-search {fasta} {db_path} {mmseq_output} {tmp_dir} --search-type 1 """ \
              f"""-e {high_eval} --threads {ncpus} --disk-space-limit {gmem}G """ \
              f"""--alignment-mode 1 --remove-tmp-files 1 && """ \
              f"""{mmseqs_path} convertalis {fasta} {db_path} {mmseq_output} {output_file} """ \
              f""" --format-output "query,target,theader,evalue" """
    os.system(command)


def run_mmseqs_on_dir_rec(dir_path, out_dir, search_db, mmseqs_path, eval, ncpus=NCPUS, gmem=GMEM, size=CONTIG_SIZE):
    files_lists = go_through_files(dir_path, size)
    for files in files_lists:
        fasta = files[0]
        output_file = os.path.join(out_dir, ".".join(os.path.split(fasta)[1].split(".")[:-1]) + "_mmseqs_res.tsv")
        if os.path.exists(output_file.replace(".tsv", "")):
            continue
        send_mmseqs_job(fasta, search_db, output_file, mmseqs_path, eval, ncpus, gmem)
