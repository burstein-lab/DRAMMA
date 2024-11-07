import numpy as np
from utilities import feature_to_file, MLFeature, get_exponent
import pandas as pd
import os
from itertools import chain
import time
import json


DB_LST = ['Bacteria_Chloroflexi_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Heterolobosea_DB_reducted', 'Bacteria_Firmicutes_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Thermodesulfobacteria_DB_reducted', 'Bacteria_Actinobacteria_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Acidobacteria_DB_reducted', 'Bacteria_Synergistia_DB_reducted', 'Bacteria_Aquificae_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Coprothermobacterota_DB_reducted', 'Bacteria_Alphaproteobacteria_DB_reducted', 'Eukaryota_Fungi_DB_reducted', 'Bacteria_Bacteroidetes_Chlorobi_group_DB_reducted', 'Bacteria_Cyanobacteria_Melainabacteria_group_DB_reducted', 'Bacteria_Betaproteobacteria_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Elusimicrobia_DB_reducted', 'Bacteria_Tenericutes_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Proteobacteria_DB_reducted', 'Bacteria_Thermotogae_DB_reducted', 'Archaea_Small_Sub_Taxes_of_Euryarchaeota_DB_reducted', 'Eukaryota_Evosea_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Haptista_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Rhizaria_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Stramenopiles_DB_reducted', 'Bacteria_Fusobacteriia_DB_reducted', 'Archaea_Small_Sub_Taxes_of_DPANN_group_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_FCB_group_DB_reducted', 'Archaea_Stenosarchaea_group_DB_reducted', 'Archaea_Crenarchaeota_DB_reducted', 'Archaea_Small_Sub_Taxes_of_TACK_group_DB_reducted', 'Bacteria_Spirochaetia_DB_reducted', 'Eukaryota_Chlorophyta_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Dictyoglomi_DB_reducted', 'Bacteria_Verrucomicrobia_DB_reducted', 'Bacteria_delta_epsilon_subdivisions_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Terrabacteria_group_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Chrysiogenetes_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Nitrospirae_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Nitrospinae_Tectomicrobia_group_DB_reducted', 'Bacteria_Gammaproteobacteria_DB_reducted', 'Archaea_unclassified_Euryarchaeota_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_PVC_group_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Deferribacteres_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Cryptophyta_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Metamonada_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Euglenozoa_DB_reducted', 'Bacteria_Small_Sub_Taxes_of_Calditrichaeota_DB_reducted', 'Archaea_Methanomada_group_DB_reducted', 'Viruses_DB_reducted', 'Eukaryota_Small_Sub_Taxes_of_Alveolata_DB_reducted']
EVALUE_THRESHOLD = 1e-6
SLEEP_TIME = 0.1
RES_COLUMNS = ['ID', 'target_id', 'target_header', 'e-value']
QUANTILES = [0.5, 0.75, 0.9]
NUM_OF_DOMAINS = 4
RETRY_NUM = 3


def remove_tmp_files(file_name):
    status_code = os.system(f"""rm -r -f "{file_name}"*""")
    if status_code != 0:
        print(f"""could not remove files starting with "{file_name}" """)


def get_row_stats(row, db_dict, e_val, total_taxes, total_dbs):
    """
    returns row stat of each row, level1 is num of domains, level2 is num of taxes (the parent tax of each db)
    and level3 is number of dbs
    """
    db_count = 0
    domains, taxes = set(), set()
    col_preffix = f'{e_val}_e-value_percentage_'
    columns = [col for col in row.index if col_preffix in col]

    for col in columns:
        val = row[col]
        if val > 0:
            db_count += 1
            db_name = col.replace(col_preffix, "")
            domains.add(db_name.split("_")[0])
            taxes.add(db_dict[db_name]["tax"])
    res = pd.Series([len(domains)/NUM_OF_DOMAINS, len(taxes)/total_taxes, db_count/total_dbs])
    return res


class MMseqsTaxonomyFeatures(MLFeature):
    def __init__(self, mmseqs_path, tax_data_path, threshold=EVALUE_THRESHOLD, ncpus=8, gmem=32, keep_files=False):
        """
        :param mmseqs_path: Path to mmseqs2 program.
        :param tax_data_path: path to directory with all relevant files for taxonmy search, shoulb be downloaded from Zenodo
        :param threshold:str/number (both work) of the max e-value of which we receive a result from mmseqs search function
        :param queue: The queue to which we send the mmseqs jobs.
        :param ncpus: The number of cpus to use for each mmseqs search run.
        :param gmem:  The amount of memory (in giga) to use for each mmseqs search run.
        :param keep_files: whether to keep the files created during the mmseqs search, and therefore be able to reuse them
        instead of executing the search again. if False, mmseqs search will be executed regardless to whether previous
        files exist or not.
        """

        self.mmseqs = mmseqs_path
        self.db_path = os.path.join(tax_data_path, 'mmseq_db', 'all_groups_united_DB_reducted_less_eukaryotes')
        self.query_dir = os.path.join(tax_data_path, 'query_dbs')
        if not os.path.exists(self.query_dir):
            os.mkdir(self.query_dir)

        self.tmp_dir = os.path.join(self.query_dir, "tmp")
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        self.dicts_dir = os.path.join(tax_data_path, "taxonomy_groups_protein_mappings")
        with open(os.path.join(tax_data_path, "taxonomy_groups_info.json"), "r") as json_f:
            self.db_info_dict = json.load(json_f)

        self.threshold = threshold
        self.ncpus = ncpus
        self.gmem = gmem
        self.keep_files = keep_files

    def __create_mmseqs_query_db(self, protein_fasta, query_db_path):
        command = f"""{self.mmseqs} createdb "{protein_fasta}" "{query_db_path}" """
        status_code = os.system(command)
        status = "Done" if status_code == 0 else "Problem Found"
        return status

    def __get_mmseqs_out_filename(self, query_db_path, res_dir):
        db_name = os.path.split(self.db_path)[1]
        query_db_name = os.path.split(query_db_path)[1]
        mmseqs_output_file = os.path.join(res_dir, f"""{db_name}_{query_db_name}_search_result.tsv""")
        return mmseqs_output_file

    def __create_search_command(self, query_db, mmseqs_output_file):
        mmseq_output = mmseqs_output_file.replace(".tsv", "")
        tmp_dir = os.path.join(self.tmp_dir, str(time.time()))

        if self.keep_files and os.path.exists(mmseqs_output_file):  # if result file exists, no need to search again
            command = f"""{self.mmseqs} convertalis {query_db} {self.db_path} {mmseq_output} {mmseqs_output_file} """ \
                      f""" --format-output "query,target,theader,evalue" """
        else:
            command = f"""{self.mmseqs} search {query_db} {self.db_path} {mmseq_output} {tmp_dir} --search-type 1 """ \
                      f"""-e {self.threshold} --threads {self.ncpus} --disk-space-limit {self.gmem}G """ \
                      f"""--alignment-mode 1 --remove-tmp-files 1 && """ \
                      f"""{self.mmseqs} convertalis {query_db} {self.db_path} {mmseq_output} {mmseqs_output_file} """ \
                      f""" --format-output "query,target,theader,evalue" """
        return command, tmp_dir

    def __run_dbs_search(self, query_db):
        dbs_dict = {}
        if not os.path.exists(self.db_path):
            print(f"""mmseqs DB {self.db_path} does not exist.""")
            return dbs_dict

        query_db_name = os.path.split(query_db)[1]
        res_dir_path = os.path.join(self.query_dir, f"_{query_db_name}") if self.keep_files else os.path.join(self.query_dir, f"{query_db_name}_{time.time()}")
        if not os.path.exists(res_dir_path):
            os.mkdir(res_dir_path)

        mmseqs_output_file = self.__get_mmseqs_out_filename(query_db, res_dir_path)
        command, tmp_dir = self.__create_search_command(query_db, mmseqs_output_file)
        os.system(command)
        return mmseqs_output_file, command, tmp_dir

    def __process_result_df_by_db(self, db_name, df):
        if len(df) == 0:  # if no results were found, so we only change the column names
            columns = [f'{self.threshold}_e-value_percentage_{db_name}', f'max_exponent_{db_name}']
            return pd.DataFrame([], columns=columns).rename_axis('ID')

        with open(os.path.join(self.dicts_dir, f'{db_name}.json'), 'r') as f_in:
            protein_dict = json.load(f_in)
        db_size = self.db_info_dict[db_name]["size"]

        # adding all the organisms with similar proteins
        df['organisms'] = df["target_id"].apply(lambda x: [] if pd.isnull(x) else protein_dict[x])
        df['exponent'] = df['e-value'].apply(get_exponent)
        df = df.groupby("ID").agg({"organisms": [
            (f'{self.threshold}_e-value_percentage_{db_name}', lambda x: len(set(chain(*list(x)))) / db_size)],
                                   'exponent': [(f'max_exponent_{db_name}', 'max')]})
        df.columns = df.columns.get_level_values(1)
        return df

    def __add_missing_cols(self, df):
        missing_cols = [f'max_exponent_{db}' for db in DB_LST if f'max_exponent_{db}' not in df.columns]
        missing_cols += [f'{self.threshold}_e-value_percentage_{db}' for db in DB_LST if f'{self.threshold}_e-value_percentage_{db}' not in df.columns]
        return df.reindex(columns=df.columns.tolist() + missing_cols)

    def __process_result_df(self, df, output_df):
        # extracting DB name from ID
        df['db_name'] = df['target_header'].apply(lambda x: x.split("|")[-1])

        groupby = df.groupby('db_name')
        df_lst = [self.__process_result_df_by_db(db_name, group_df) for db_name, group_df in groupby]
        out_df = output_df.join(df_lst, how='outer')
        return out_df

    def __process_result_file(self, res_file, output_df):
        # process search result file into a DF and joins it with the DF of the previous results
        if not os.path.exists(res_file):
            print("failed creating result file: " + res_file)
            return
        df = pd.read_csv(res_file, sep="\t", names=RES_COLUMNS)
        try_num = 0
        while len(df) == 0 and try_num < 10:
            try_num += 1
            time.sleep(1)
            df = pd.read_csv(res_file, sep="\t", names=RES_COLUMNS)

        output_df = self.__process_result_df(df, output_df)

        if not self.keep_files:  # removing all the result files created during the search (as there are a lot of them)
            remove_tmp_files(res_file.replace(".tsv", "."))
        return output_df

    def __process_run(self, res_file, command, output_df):
        retry_count = 0
        while not os.path.exists(res_file) and retry_count < RETRY_NUM:  # giving another chance to failed file
            print(f"failed searching against target DB")
            retry_count += 1
            time.sleep(SLEEP_TIME)
            os.system(command)
        if not os.path.exists(res_file):  # Out of retries
            print(f"Stopped retries. Failed Mmseqs search of following command: {command}. ")
            return pd.DataFrame([], columns=['index'])

        df = self.__process_result_file(res_file, output_df)
        return df

    def __get_search_results_df(self, fasta_file_name, query_db, ids_df):
        output_df = ids_df.set_index("ID") if ids_df is not None else pd.DataFrame([]).rename_axis('ID')

        # running mmseqs search against the united DB
        mmseqs_output_file, command, tmp_dir = self.__run_dbs_search(query_db)
        # process the db result file
        output_df = self.__process_run(mmseqs_output_file, command, output_df)
        os.system(f"rm -r {tmp_dir}")

        # making 'ID' into column instead of Index and filling empty values with zeros
        output_df = output_df.reset_index().rename(columns={"index": "ID"}).fillna(0)
        output_df = self.__add_missing_cols(output_df)
        return output_df

    def __add_general_features(self, df):
        exponent_cols = [col for col in df.columns if "max_exponent" in col]
        total_taxes = len({v["tax"] for v in self.db_info_dict.values()})
        col_names = [f"level_1_{self.threshold}_percentage", f"level_2_{self.threshold}_percentage", f"level_3_{self.threshold}_percentage"]

        if len(df) > 0:
            df[col_names] = df.apply(get_row_stats, axis=1, args=(self.db_info_dict, self.threshold, total_taxes, len(self.db_info_dict)))
        else:
            df = df.reindex(columns=df.columns.tolist() + col_names, fill_value=0.0)

        for q in QUANTILES:
            df[f"{q}_quantile_exponent"] = df[exponent_cols].apply(lambda r: np.nanquantile(r, q), axis=1)

        return df

    def get_features(self, protein_fasta, ids=None):
        """
        This function runs the whole script. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param ids: a dataframe containing all of the fasta's IDs
        :return: output dataframe with MMseqs features.
        """
        # query_db = self.query_dir/fasta_file_name_QueryDB
        query_db = os.path.join(self.query_dir, ".".join(os.path.split(protein_fasta)[1].split(".")[:-1]) + "_QueryDB")
        status = self.__create_mmseqs_query_db(protein_fasta, query_db)
        if status != "Done":
            print("Error occurred while creating mmseqs query DB!")
            return pd.DataFrame([]).rename_axis('ID')

        output_df = self.__get_search_results_df(protein_fasta, query_db, ids)
        output_df = self.__add_general_features(output_df)

        # removing files related to the mmseqs query DB that was created and tmp files created by mmseqs
        if not self.keep_files:
            remove_tmp_files(query_db)

        if "level_0" in output_df.columns:
            output_df = output_df.drop(columns=["ID"]).rename(columns={"level_0": "ID"})
            output_df = output_df.set_index('ID').astype('float16')
            missing_ids = [i for i in ids['ID'] if i not in output_df.index] if ids is not None else []
            output_df = output_df.reindex(index=output_df.index.tolist() + missing_ids, fill_value=0.0)
        else:
            output_df = output_df.set_index('ID').astype('float16')
        return output_df

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves mmseqs features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, data: not used by this func, only accepted because this is an abstract method
        :param ids: a dataframe containing all of the fasta's IDs
        :param out_dir: path to output directory
        :return: output dataframe with MMseqs features.
        """
        feature_to_file('mmseq_prox', dir_path=out_dir)(self.get_features)(protein_fasta, ids)
