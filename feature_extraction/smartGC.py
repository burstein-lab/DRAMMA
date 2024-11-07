import pandas as pd
import numpy as np
from utilities import k_Mers_Count, feature_to_file, MISSING_VALUE_SYMBOL, MLFeature

MIN_GENES_ON_CONTIG = 5
FEATURE_NAME = 'SmartGC_diff_from_AVG'


class SmartGCFeatures(MLFeature):
    def __init__(self, smart_dict):
        self.smart_gc, self.smart_at = SmartGCFeatures.__get_smart_GC_dictionaries(smart_dict)

    @staticmethod
    def __get_smart_GC_dictionaries(dictfile):
        '''
        we want to have the dictionary uploaded so we can change it later as we please.
        the dictionary is where we store the GC value of the amino acids.
        :param dictfile: path to the txt file with the information on the gc and at count, organized (codon GC AT)
        :return: 2 dictionaries, one that shows the GC value of the amino acid and one for the AT value.
        '''
        smart_gc, smart_at = {}, {}
        with open(dictfile, mode='r') as in_file:
            for line in in_file:
                codon, gc, ta = line.strip().split(' ', 2)
                smart_gc[codon], smart_at[codon] = int(gc.strip()), int(ta.strip())
        return smart_gc, smart_at

    def __get_single_smart_GC(self, seq):
        '''
        checks the Smart GC count of the sequence and returns its value - how many time the gene 'picked' G/C instead of A/T.
        :param seq: string of nucleotides, devidable by 3 (full codons)
        :return: the ratio of the smart GC count.
        '''
        codon_count = dict(k_Mers_Count(3, seq, 3, is_dna=True))
        GC, AT = 0, 0
        for codon, count in codon_count.items():
            GC += self.smart_gc[codon] * count
            AT += self.smart_at[codon] * count
        total = GC + AT
        if total == 0:
            print('Its an empty gene!!')
            return MISSING_VALUE_SYMBOL
        return GC / total

    def __get_smart_GC_list(self, genelist):
        '''
        wraps the single GC to give the contigs smartgc ratio of each gene.
        :return: a dataframe with the ids and the smartGC ratio of the genes.
        '''
        results_list = []
        if len(genelist) < MIN_GENES_ON_CONTIG:
            if len(genelist) == 0:
                print('we are not suppose to have an empty list!')

            results_list = [{'ID': gene.id, FEATURE_NAME: MISSING_VALUE_SYMBOL} for gene in genelist]
            df = pd.DataFrame.from_records(results_list)
            return df

        total_smart_count = [(gene.id, self.__get_single_smart_GC(str(gene.seq))) for gene in genelist]
        total_smart_count = [single_gc for single_gc in total_smart_count if single_gc[1] != MISSING_VALUE_SYMBOL]
        gc_sum = sum([single_gc[1] for single_gc in total_smart_count])
        for gene_id, single_smart_gc in total_smart_count:
            local_mean = (gc_sum - single_smart_gc) / (len(total_smart_count) - 1)  # the mean without one gene
            updated_single_smart_gc = np.abs(single_smart_gc - local_mean)
            results_dict = {'ID': gene_id, FEATURE_NAME: updated_single_smart_gc}
            results_list.append(results_dict)
        df = pd.DataFrame.from_records(results_list)
        return df

    def get_features(self, fa, genelist):
        '''
        runs the smartGC, need the dictionaries
        :param fa: unused by this func, is only sent because of run_feature_to_file
        :param genelist: list of biopython SeqIO of the genes.
        :return: a dataframe with the ids and the smartGC ratio of the genes.
        '''
        df_lst = [self.__get_smart_GC_list(gene) for gene in genelist]
        final_df = pd.concat(df_lst, ignore_index=True)
        return final_df.set_index('ID').astype("float16")

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves smart_GC features to files. Parameters are given from the user and from class instance.
        :param protein_fasta, gff, ids: not used by this func, only accepted because this is an abstract method
        :param fa: only sent to get_features so feature to file will work, as a filename is needed.
        :param data: a dataframe containing contig name, seq, genes list
        :param out_dir: path to output directory
        """
        feature_to_file('Smart_GC', dir_path=out_dir)(self.get_features)(fa, data['genes_list'])
