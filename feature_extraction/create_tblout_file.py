import subprocess
import os
import time


def create_file_name(hmm_file_name, fasta_file):
    hmm_parse = os.path.splitext(os.path.basename(hmm_file_name))[0]
    faa_parse = os.path.basename(fasta_file).split('.contigs.')[0]
    file_name = f'{faa_parse}.VS.{hmm_parse}.tblout'
    return file_name


def is_valid_tblout(tblout_path):
    print("in is_valid")
    time.sleep(10)
    with open(tblout_path, 'r') as in_file:
        lines = in_file.readlines()
    return len(lines) > 0 and lines[-1].strip() == '# [ok]'  # File finished successfully


def get_tblout_file(hmmer_path, hmm_file_name, fasta_file, retry=3, is_domain=False, tblout_path="", cpu=3):
    option = "--domtblout" if is_domain else '--tblout'
    tblout_path = tblout_path if tblout_path else os.path.join(os.getcwd(), create_file_name(hmm_file_name, fasta_file))
    if not os.path.isfile(tblout_path) or not is_valid_tblout(tblout_path):
        retry_num = 0
        while retry_num < retry:
            print(f'hmmsearch --cpu {cpu-1} {option} {tblout_path} {hmm_file_name} {fasta_file}') # cpu -1 since there is a master thread that is not counted by this parameter
            sp = subprocess.run(f'{hmmer_path} -o /dev/null --cpu {cpu-1} {option} {tblout_path} {hmm_file_name} {fasta_file}', shell=True)
            if sp.returncode != 0:
                print(f"Failed running hmmsearch of {fasta_file} on {hmm_file_name}.")
                if os.path.exists(tblout_path):
                    os.remove(tblout_path)
                raise Exception(f"HMM search failed for {fasta_file} with errorcode: {sp.returncode}")
            if not is_valid_tblout(tblout_path):
                retry_num += 1
                print(f"retry_num: {retry_num}")
                os.rename(tblout_path, tblout_path.replace(".tblout", f"_retry_{retry_num}.tblout"))
                if retry_num == retry: # no more tries
                    raise Exception(f"Tblout could not be written correctly for {fasta_file}")
            else:
                print("finished retries")
                break
    return tblout_path

