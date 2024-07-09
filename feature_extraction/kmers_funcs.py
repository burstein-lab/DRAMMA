from itertools import product
import pandas as pd
import scipy.spatial.distance as sp
from Bio.Seq import reverse_complement
from utilities import MISSING_VALUE_SYMBOL, feature_to_file, CONTIG_GENE_RATIO, k_Mers_Count, MLFeature


STEP = 1


def get_k_Mers_list(k):
    '''
    creates a list of all the possible combinations, so we won't miss one that doesn't exists in the gene
    :param k: the k of the k-mers
    :return: a list of all the possible combinations
    '''
    perm = product(['A', 'C', 'G', 'T'], repeat=k)
    return [''.join(i) for i in list(perm)]


class KMersFeatures(MLFeature):
    def __init__(self, kmer_size):
        self.max_k = kmer_size  # maximal k to check
        self.canonical_dicts = {k: KMersFeatures.__create_canonical_dict(k) for k in range(2, self.max_k + 1)}

    @staticmethod
    def __create_canonical_dict(k):
        '''
        creates an empty dictionary with all the options, and also a canonical dictionary that will be used to hash the
        kmers to the right place. meaning we are taking into account that the reverse_complement is the same in meta
        because we don't know the reading direction, so for ex. AA=TT, AAC = GTT
        :param k: the k of the k-mers
        :return: 2 dictionaries, first for the hash function, the second to have all the kmers at 0.
        '''
        canocical_dict = {}
        for kmer in get_k_Mers_list(k):
            canocical_dict[kmer] = canocical_dict.get(reverse_complement(kmer), kmer)  # Makes sure we only use one of the RC
        return canocical_dict

    def __unite_RC_kmers(self, counted_kmers, canocical_dict):
        """Given the Kmers Counter and the canonical dict mapping the RCs, unite RCs count and fill missing kmers with 0"""
        for key, value in canocical_dict.items():
            if key != value:  # we don't keep this Kmer
                counted_kmers[value] += counted_kmers[key]
                del counted_kmers[key]
            else:  # to make sure all kmers are in counter
                counted_kmers[value] += 0
        return counted_kmers

    def __fill_res_dict(self, gene, k, euclidean_distance, correlation_distance, cosine_distance):
        results_dict = {}
        results_dict['ID'] = gene.id
        results_dict[f'euclidean_distance_{k}mers_nuc'] = euclidean_distance
        results_dict[f'correlation_distance_{k}mers_nuc'] = correlation_distance
        results_dict[f'cosine_distance_{k}mers_nuc'] = cosine_distance
        return results_dict

    def get_gene_contig_distance(self, gene, k, canonical_dict, contig_kmers_num, contigVec):
        geneseq = str(gene.seq)
        gene_kmers_num = len(geneseq) - (k - 1)
        local_contig_kmers_num = contig_kmers_num - gene_kmers_num

        if local_contig_kmers_num / gene_kmers_num < CONTIG_GENE_RATIO:  # make sure the contig size is significant
            return self.__fill_res_dict(gene, k, MISSING_VALUE_SYMBOL, MISSING_VALUE_SYMBOL, MISSING_VALUE_SYMBOL)

        genemersCounter = k_Mers_Count(k, geneseq, STEP, is_dna=True)
        geneVec = self.__unite_RC_kmers(genemersCounter, canonical_dict)
        localcontigvec = contigVec.copy()
        localcontigvec.subtract(geneVec)

        # updating dicts to store percentage instead of count
        geneVec = {kmer: float(count / gene_kmers_num) for kmer, count in geneVec.items()}
        localcontigvec = {kmer: float(count / local_contig_kmers_num) for kmer, count in localcontigvec.items()}

        # sorting kmers_percentage by kmers so vectors will be aligned
        geneVec_values = [items[1] for items in sorted(geneVec.items(), key=lambda x: x[0])]
        localcontigvec_values = [items[1] for items in sorted(localcontigvec.items(), key=lambda x: x[0])]
        correlation_distance = sp.correlation(geneVec_values, localcontigvec_values)
        cosine_distance = sp.cosine(geneVec_values, localcontigvec_values)
        euclidean_distance = sp.euclidean(geneVec_values, localcontigvec_values)
        return self.__fill_res_dict(gene, k, euclidean_distance, correlation_distance, cosine_distance)

    def __k_mers_Vec_Calc(self, contigseq, geneseqlist, k):
        '''
        calculates the ratio of each contig and genes k mers.
        :param contigseq: the sequence of the entire contig
        :param geneseqlist: a list of all the sequences (ORFs) in the contig
        :param k: the 'k' mer to calc with.
        :return: dataframe with the ID of the genes and the wanted distances calculated
        '''
        contigseq = str(contigseq)
        canonical_dict = self.canonical_dicts[k]
        contigCounter = k_Mers_Count(k, contigseq, STEP, is_dna=True)
        contigVec = self.__unite_RC_kmers(contigCounter, canonical_dict)
        contig_kmers_num = len(contigseq) - (k - 1)
        results_list = [self.get_gene_contig_distance(gene, k, canonical_dict, contig_kmers_num, contigVec) for gene in geneseqlist]
        return pd.DataFrame.from_records(results_list)

    def get_features(self, fa, data):
        '''
       Gets the dna kmers, for all kmers up to kmer_size
        :param data: df, contig name, seq, genes list
        :param fa: unused by this func, is only sent because of run_feature_to_file
        :return: a dataframe with the id and the distances between the differente k's and the contig.
        '''
        finaldataframe = pd.DataFrame(columns=['ID'])
        if len(data) == 0:
            print('Data is empty, cannot calculate Kmers features!')
            print(f'fa file is: {fa}')
            return pd.DataFrame([]).rename_axis('ID')
        for k in range(2, self.max_k + 1):
            listoresults = data.apply(lambda r: self.__k_mers_Vec_Calc(r['contig_seq'], r['genes_list'], k), axis=1).tolist()
            curr_df = pd.concat(listoresults, ignore_index=True, sort=False)
            finaldataframe = pd.merge(finaldataframe, curr_df, on='ID', how='outer')
        finaldataframe.set_index('ID', inplace=True)
        return finaldataframe.astype('float32')

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves Kmers features to files. Parameters are given from the user and from class instance.
        :param protein_fasta, gff, ids: not used by this func, only accepted because this is an abstract method
        :param fa: only sent to get_features so feature to file will work, as a filename is needed.
        :param data: a dataframe containing contig name, seq, genes list
        :param out_dir: path to output directory
        """
        feature_to_file('DNA_Kmers', dir_path=out_dir)(self.get_features)(fa, data)
