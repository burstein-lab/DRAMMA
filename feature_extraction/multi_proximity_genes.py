import pandas as pd
import os
from .create_tblout_file import get_tblout_file
from .create_proximity_df import calculate_gene_proximity
from utilities import feature_to_file, MLFeature


FILE_TO_DB_NAME = {'Acrs': 'Acrs_HMM', 'Antirestriction_genes': 'Antirestriction_HMM_merged_version4', 'TnpPred.uniq': 'TnpPred.uniq',
                   'pfam_phage_plasmid_proteins': 'pfam_phage_plasmid_proteins', 'T6SS.immunity.proteins': 'T6SS.immunity.proteins',
                   'VirSorter_Phage_Clusters_hallmark': 'VirSorter_Phage_Clusters_current.hallmark.annot', 'DRAMMA_ARG_DB': 'all_ARGs_filtered'}


class MultiProximityFeatures(MLFeature):
    def __init__(self, hmmer_path, hmm_dir, threshold_list, by_eval, gene_window=5, nucleotide_window=5000, delete_files=False):
        self.hmmer_path = hmmer_path
        self.hmm_dir = hmm_dir
        self.threshold_list = threshold_list
        self.by_eval = by_eval
        self.gene_window = gene_window
        self.nucleotide_window = nucleotide_window
        self.delete_files = delete_files

    def __remove_unnecessary_columns(self, df) -> pd.DataFrame:
        windows = df[df.filter(like='genes_in').columns]
        nucleotide_distance = df[df.filter(like='nucleotide_distance').columns]
        gene_distance = df[df.filter(like='gene_distance').columns]

        return windows.join([nucleotide_distance, gene_distance])

    def __fill_missing_values(self, full_df):
        # filling missing gene distance values with default value of  gene_window * 2
        full_df[full_df.filter(like='gene_distance').columns] = full_df[full_df.filter(like='gene_distance').columns].fillna(self.gene_window * 2)
        # filling missing nucleotide distance values with default value of  nucleotide_distance * 2
        full_df[full_df.filter(like='nucleotide_distance').columns] = full_df[full_df.filter(like='nucleotide_distance').columns].fillna(self.nucleotide_window * 2)
        # everything else is filled with zeros
        return full_df.fillna(0)

    def get_features(self, proteins_fasta, gff, ids_df):
        df_list = [ids_df.set_index("ID")]
        for hmm_db in os.listdir(self.hmm_dir):
            hmm_db_name = FILE_TO_DB_NAME.get(hmm_db.replace('.hmm', ''), hmm_db.replace('.hmm', ''))
            hmm_file = os.path.join(self.hmm_dir, hmm_db)
            tblout = get_tblout_file(self.hmmer_path, hmm_file, proteins_fasta)

            one_df = calculate_gene_proximity(gff, tblout, hmm_db_name, self.threshold_list, self.by_eval,
                                                  self.gene_window, self.nucleotide_window)
            clean_df = self.__remove_unnecessary_columns(one_df)
            df_list.append(clean_df)
            if self.delete_files:
                os.remove(tblout)
        full_df = df_list[0].join(df_list[1:])
        full_df = self.__fill_missing_values(full_df)
        full_df = full_df.apply(pd.to_numeric, downcast='integer')  # reduces memory
        return full_df

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves multi_proximity features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff: a gff file
        :param fa, data: not used by this func, only accepted because this is an abstract method
        :param ids: a dataframe containing all of the fasta's IDs
        :param out_dir: path to output directory
        """
        feature_to_file(f'HMM_Proximity_{self.gene_window}', dir_path=out_dir)(self.get_features)(protein_fasta, gff, ids)
