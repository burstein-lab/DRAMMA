import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from utilities import feature_to_file, MLFeature


COLS = ['ID', 'molecular_weight_pp', 'aromaticity_pp', 'gravy_pp', 'instability_index_pp', 'isoelectric_point_pp', 'Helix_fraction_pp', 'Turn_fraction_pp', 'Sheet_fraction_pp', 'molar_extinction_coefficient_reduced_pp', 'molar_extinction_coefficient_cystines_pp']


class ProtParamsFeatures(MLFeature):
    @staticmethod
    def __get_params(record):
        protein_seq = (str(record.seq)).replace('*', '')
        p = ProteinAnalysis(protein_seq)
        helix, turn, sheet = p.secondary_structure_fraction()
        molar_coefficient_reduced, molar_coefficient_cystines = p.molar_extinction_coefficient()
        try:
            param_dict = {'ID': record.id, 'molecular_weight_pp': p.molecular_weight(), 'aromaticity_pp': p.aromaticity(), 'gravy_pp': p.gravy(),
                          'instability_index_pp': p.instability_index(), 'isoelectric_point_pp': p.isoelectric_point(),
                          'Helix_fraction_pp': helix, 'Turn_fraction_pp': turn, 'Sheet_fraction_pp': sheet,
                          'molar_extinction_coefficient_reduced_pp': molar_coefficient_reduced, 'molar_extinction_coefficient_cystines_pp': molar_coefficient_cystines}
        except (ValueError, KeyError):
            print(f'Problem with ProtParams Features on {record.id}')
            param_dict = {'ID': record.id, 'molecular_weight_pp': np.nan,
                          'aromaticity_pp':np.nan, 'gravy_pp': np.nan,
                          'instability_index_pp': np.nan, 'isoelectric_point_pp': np.nan,
                          'Helix_fraction_pp': helix, 'Turn_fraction_pp': turn, 'Sheet_fraction_pp': sheet,
                          'molar_extinction_coefficient_reduced_pp': molar_coefficient_reduced,
                          'molar_extinction_coefficient_cystines_pp': molar_coefficient_cystines}
        return pd.Series(param_dict)[COLS]  # to keep order of columns

    def get_features(self, protein_file):
        df = pd.DataFrame()
        df['Record'] = pd.Series(SeqIO.parse(protein_file, "fasta"))
        df[COLS] = df['Record'].apply(ProtParamsFeatures.__get_params)
        return df.set_index('ID').drop(columns=['Record'])

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves protein params features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, ids, data: not used by this func, only accepted because this is an abstract method
        :param out_dir: path to output directory
        """
        feature_to_file('Prot_param', dir_path=out_dir)(self.get_features)(protein_fasta)
