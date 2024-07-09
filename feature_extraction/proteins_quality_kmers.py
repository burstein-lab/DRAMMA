from Bio import SeqIO
import pandas as pd
from functools import reduce
from collections import Counter
from utilities import k_Mers_Count, feature_to_file, MLFeature

#  * represents stop codon
HYDRO_DICT = {'S': 'L', 'T': 'L', 'N': 'L', 'K': 'L', 'Y': 'L', 'E': 'L', 'Q': 'L', 'H': 'L', 'D': 'L', 'R': 'L',
              'Z': 'L', 'B': 'L', 'A': 'B', 'G': 'B', 'I': 'B', 'L': 'B', 'M': 'B', 'V': 'B', 'P': 'B', 'F': 'B',
              'W': 'B', 'C': 'B', '*': '*'}


class SmartAAKmersFeatures(MLFeature):
    def __init__(self, min_k, max_k):
        self.min_k = min_k
        self.max_k = max_k

    @staticmethod
    def __transformAA(seq):
        '''
        transforms the AA sequence to a more condense representation that will hopefully give new insights.
        based on this paper: https://peerj.com/articles/7055/ with the change of moving P,F,W,C to hydrophobic group
        '''
        transformed = []
        for amino_acid in seq:
            if amino_acid in HYDRO_DICT:
                transformed.append(HYDRO_DICT[amino_acid])
            else:
                print('problem with unknown amino acids ' + amino_acid)

        return ''.join(transformed).split('*')  # separating seqs by stop codons

    @staticmethod
    def __runSmartAAKmers(protein_fasta, k):
        results_list = []
        for record in SeqIO.parse(protein_fasta, "fasta"):
            newAAs = SmartAAKmersFeatures.__transformAA(record.seq)
            AAkmerscount = Counter()
            for newAA in newAAs:
                AAkmerscount += k_Mers_Count(k, newAA, 1)
            n = sum(AAkmerscount.values())
            AAkmerscount = {k: round((count / n) * 100, 2) for k, count in AAkmerscount.items()}
            AAkmerscount.update({'ID': record.id})
            results_list.append(AAkmerscount)
        return pd.DataFrame.from_records(results_list)

    def get_features(self, protein_fasta):
        dataframeslist = [SmartAAKmersFeatures.__runSmartAAKmers(protein_fasta, k) for k in range(self.min_k, self.max_k + 1)]
        df = reduce(lambda left, right: pd.merge(left, right, on='ID'), dataframeslist)
        df = df.fillna(0).set_index('ID')
        return df.astype('float16')

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves Smart AA Kmers features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, data, ids: not used by this func, only accepted because this is an abstract method
        :param out_dir: path to output directory
        """
        feature_to_file('Smart_AA_Kmers', dir_path=out_dir)(self.get_features)(protein_fasta)
