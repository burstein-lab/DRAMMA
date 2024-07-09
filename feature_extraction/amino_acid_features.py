from collections import Counter
from Bio import SeqIO
import pandas as pd
import numpy as np
import aaIndex
from utilities import feature_to_file, MLFeature

AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# the different groups from Wikipedia
amino_groups = {'Aliphatic[AGILV]': ['A', 'G', 'I', 'L', 'V'], 'Amide[NQ]': ['N', 'Q'], "Aromatic[FWY]": ['F', 'W', 'Y'],
                'Hydroxyl-Containing[ST]': ['S', 'T'], 'Sulfur-Containing[CM]': ['C', 'M'], 'Negative[DE]': ['D', 'E'],
                'Nonpolar[AGILVFWPCM]': ['A', 'G', 'I', 'L', 'V', 'F', 'W', 'P', 'C', 'M'], 'Positive[KR]': ['K', 'R'],
                'Neutral[HAGILVFWPCMNQYST]': ['H', 'A', 'G', 'I', 'L', 'V', 'F', 'W', 'P', 'C', 'M', 'N', 'Q', 'Y', 'S','T'],
                'Polar[NQYST]': ['N', 'Q', 'Y', 'S', 'T']}


class AAFeatures(MLFeature):
    def __init__(self, index_file):
        self.index_list = AAFeatures.__get_index_list(index_file)

    @staticmethod
    def __get_index_list(index_file):
        aaIndex.init(path='.')
        with open(index_file, 'r') as f:
            indexes = f.read().splitlines()
            lines = [line.split()[0] for line in indexes]
            index_list = [(line, aaIndex.get(line)) for line in lines]
        return index_list

    @staticmethod
    def __get_seq_aa_percentage(record, aa_zero_dict):
        # calculates the proportion of each amino acid in the Fasta sequence
        fasta_aa_count = Counter(record)
        fasta_aa_count.pop('*', None)
        fasta_aa_count.update(aa_zero_dict)
        n = sum(fasta_aa_count.values())
        aa_props = {k: round((count / n) * 100, 2) for k, count in fasta_aa_count.items()}
        aa_props['ID'] = record.id
        return aa_props

    @staticmethod
    def __calc_aa_percentage(protein_file):
        aa_zero_dict = {aa: 0 for aa in AA_LIST}
        listOfProt = [AAFeatures.__get_seq_aa_percentage(rec, aa_zero_dict) for rec in SeqIO.parse(protein_file, "fasta")]
        result = pd.DataFrame.from_records(listOfProt)
        return result.set_index('ID').astype("float16")

    @staticmethod
    def __get_aa_index_per_index(index_name, index, aa_per, amino_acids):
        # receives an index and index_name and returns a series with the relevant aa_index information
        index_values = np.array([index.get(oneamino) for oneamino in amino_acids])
        index_values = np.where(pd.isnull(index_values), 0.0, index_values)
        aa_res = aa_per.mul(index_values)  # for each amino acid multiply percentage with value from index
        aa_res = aa_res.sum(axis=1) / 100  # properties for entire gene and not per AA
        return aa_res.rename(index_name)

    def __aaIndexCalc(self, aa_per):
        amino_acids = tuple(aa_per.keys())
        aa_index_results = [AAFeatures.__get_aa_index_per_index(ind_name, ind, aa_per, amino_acids) for (ind_name, ind) in self.index_list]
        finaldf = pd.concat(aa_index_results, axis=1).reset_index().rename(columns={'index': 'ID'})
        return finaldf.set_index('ID').astype("float16")

    @staticmethod
    def __runFastaAaQualities(aa_per):
        # calculates the proportion of the amino acid groups.
        # the input is a panda Dataframe with the proportions of each amino acid.
        qualDf = pd.DataFrame()
        for group_name, aminos in amino_groups.items():
            qualDf[group_name] = aa_per[aminos].sum(axis=1)
        qualDf = qualDf.reset_index().rename(columns={'index': 'ID'})  # important for production later
        return qualDf.set_index('ID').astype("float16")

    def get_features(self, fasta_file):
        aa_per = AAFeatures.__calc_aa_percentage(fasta_file)
        aa_index = self.__aaIndexCalc(aa_per)
        aa_qualities = AAFeatures.__runFastaAaQualities(aa_per)
        df = pd.concat([aa_per, aa_index, aa_qualities], axis=1)
        return df

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves amino acid features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, ids, data: not used by this func, only accepted because this is an abstract method
        :param out_dir: path to output directory
        """
        feature_to_file('AA_features', dir_path=out_dir)(self.get_features)(protein_fasta)
